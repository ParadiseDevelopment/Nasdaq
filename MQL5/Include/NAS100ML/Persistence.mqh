//+------------------------------------------------------------------+
//|                                                  Persistence.mqh |
//|  Plain-text model checkpoints.                                   |
//|                                                                  |
//|  One number per line so the file can be written or read by the   |
//|  offline trainer in tools/train_offline.py without any binary    |
//|  layout assumptions. Layout:                                     |
//|                                                                  |
//|      #NAS100ML                                                   |
//|      VERSION 2                                                   |
//|      NFEAT <n>                                                   |
//|      SECTION SCALER <count>                                      |
//|      ... <count> lines ...                                       |
//|      SECTION LOGISTIC <count>                                    |
//|      SECTION RFF <count>                                         |
//|      SECTION MLP <count>                                         |
//|      SECTION HEDGE 3                                             |
//|      END                                                         |
//+------------------------------------------------------------------+
#ifndef __NAS100ML_PERSISTENCE_MQH__
#define __NAS100ML_PERSISTENCE_MQH__

#include "Utils.mqh"
#include "Scaler.mqh"
#include "Ensemble.mqh"
#include "FeatureEngine.mqh"

#define NASML_CHECKPOINT_VERSION 2

//+------------------------------------------------------------------+
void NasmlWriteSection(const int fh, const string tag, const double &v[])
  {
   int n = ArraySize(v);
   FileWriteString(fh, "SECTION " + tag + " " + IntegerToString(n) + "\r\n");
   for(int i = 0; i < n; i++)
      FileWriteString(fh, StringFormat("%.12g", v[i]) + "\r\n");
  }

//+------------------------------------------------------------------+
//| Save scaler + all three experts + hedge weights.                 |
//+------------------------------------------------------------------+
bool NasmlSaveCheckpoint(const string filename, CScaler &scaler, CEnsemble &ens, const int nfeat)
  {
   int fh = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      PrintFormat("NASML: cannot open '%s' for writing (error %d)", filename, GetLastError());
      return(false);
     }

   FileWriteString(fh, "#NAS100ML\r\n");
   FileWriteString(fh, "VERSION " + IntegerToString(NASML_CHECKPOINT_VERSION) + "\r\n");
   FileWriteString(fh, "NFEAT " + IntegerToString(nfeat) + "\r\n");

   double buf[];
   scaler.GetParams(buf);          NasmlWriteSection(fh, "SCALER",   buf);
   ens.Linear().GetParams(buf);    NasmlWriteSection(fh, "LOGISTIC", buf);
   ens.Rff().GetParams(buf);       NasmlWriteSection(fh, "RFF",      buf);
   ens.Mlp().GetParams(buf);       NasmlWriteSection(fh, "MLP",      buf);
   ens.GetHedge(buf);              NasmlWriteSection(fh, "HEDGE",    buf);

   FileWriteString(fh, "END\r\n");
   FileClose(fh);
   return(true);
  }

//+------------------------------------------------------------------+
//| Load a checkpoint written either by the EA or by the offline     |
//| trainer. Unknown sections are skipped so the format can grow.    |
//+------------------------------------------------------------------+
bool NasmlLoadCheckpoint(const string filename, CScaler &scaler, CEnsemble &ens, const int nfeat)
  {
   if(!FileIsExist(filename))
     {
      PrintFormat("NASML: checkpoint '%s' not found", filename);
      return(false);
     }

   int fh = FileOpen(filename, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      PrintFormat("NASML: cannot open '%s' for reading (error %d)", filename, GetLastError());
      return(false);
     }

   string header = FileReadString(fh);
   if(StringFind(header, "#NAS100ML") < 0)
     {
      Print("NASML: checkpoint header not recognised");
      FileClose(fh);
      return(false);
     }

   bool okScaler = false, okLin = false, okRff = false, okMlp = false, okHedge = false;
   int  fileNfeat = -1;

   while(!FileIsEnding(fh))
     {
      string line = FileReadString(fh);
      StringTrimLeft(line);
      StringTrimRight(line);
      if(StringLen(line) == 0)
         continue;
      if(line == "END")
         break;

      string parts[];
      int k = StringSplit(line, StringGetCharacter(" ", 0), parts);

      if(k >= 2 && parts[0] == "NFEAT")
        {
         fileNfeat = (int)StringToInteger(parts[1]);
         if(fileNfeat != nfeat)
           {
            PrintFormat("NASML: checkpoint has %d features, EA expects %d - ignoring file",
                        fileNfeat, nfeat);
            FileClose(fh);
            return(false);
           }
         continue;
        }

      if(k >= 3 && parts[0] == "SECTION")
        {
         string tag = parts[1];
         int    cnt = (int)StringToInteger(parts[2]);
         if(cnt < 0 || cnt > 5000000)
           {
            Print("NASML: implausible section size, aborting load");
            FileClose(fh);
            return(false);
           }

         double v[];
         ArrayResize(v, cnt);
         for(int i = 0; i < cnt; i++)
           {
            if(FileIsEnding(fh))
              {
               Print("NASML: checkpoint truncated inside section " + tag);
               FileClose(fh);
               return(false);
              }
            string s = FileReadString(fh);
            StringTrimLeft(s);
            StringTrimRight(s);
            v[i] = StringToDouble(s);
           }

         if(tag == "SCALER")        okScaler = scaler.SetParams(v);
         else if(tag == "LOGISTIC") okLin    = ens.Linear().SetParams(v);
         else if(tag == "RFF")      okRff    = ens.Rff().SetParams(v);
         else if(tag == "MLP")      okMlp    = ens.Mlp().SetParams(v);
         else if(tag == "HEDGE")    okHedge  = ens.SetHedge(v);
         // any other tag: silently ignored
        }
     }

   FileClose(fh);

   PrintFormat("NASML: checkpoint loaded  scaler=%s logistic=%s rff=%s mlp=%s hedge=%s",
               okScaler ? "ok" : "-", okLin ? "ok" : "-", okRff ? "ok" : "-",
               okMlp ? "ok" : "-", okHedge ? "ok" : "-");

   return(okScaler || okLin || okRff || okMlp);
  }

//+------------------------------------------------------------------+
//| Dataset export - one CSV row per resolved training sample.       |
//+------------------------------------------------------------------+
class CDatasetWriter
  {
private:
   int               m_fh;
   int               m_nfeat;
   long              m_rows;

public:
                     CDatasetWriter(void) : m_fh(INVALID_HANDLE), m_nfeat(0), m_rows(0) {}
                    ~CDatasetWriter(void) { Close(); }

   bool              Open(const string filename, const int nfeat)
     {
      m_nfeat = nfeat;
      m_fh = FileOpen(filename, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(m_fh == INVALID_HANDLE)
        {
         PrintFormat("NASML: cannot open dataset '%s' (error %d)", filename, GetLastError());
         return(false);
        }
      string header = "time";
      for(int i = 0; i < nfeat; i++)
         header += "," + CFeatureEngine::FeatureName(i);
      header += ",label,weight";
      FileWriteString(m_fh, header + "\r\n");
      m_rows = 0;
      return(true);
     }

   void              Write(const datetime t, const double &x[], const double y, const double w)
     {
      if(m_fh == INVALID_HANDLE)
         return;
      string row = TimeToString(t, TIME_DATE | TIME_MINUTES);
      for(int i = 0; i < m_nfeat; i++)
         row += "," + StringFormat("%.8g", x[i]);
      row += "," + StringFormat("%.0f", y) + "," + StringFormat("%.4f", w);
      FileWriteString(m_fh, row + "\r\n");
      m_rows++;
     }

   long              Rows(void) const { return(m_rows); }

   void              Close(void)
     {
      if(m_fh != INVALID_HANDLE)
        {
         FileClose(m_fh);
         m_fh = INVALID_HANDLE;
        }
     }
  };

#endif // __NAS100ML_PERSISTENCE_MQH__
//+------------------------------------------------------------------+
