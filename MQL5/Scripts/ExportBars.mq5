//+------------------------------------------------------------------+
//|                                                   ExportBars.mq5 |
//|                                                                  |
//|  Dumps the current chart's bars to CSV so tools/backtest.py can   |
//|  replay them.                                                     |
//|                                                                  |
//|  Usage: drag onto a NAS100 chart of the timeframe you want, set   |
//|  InpBars (M15 for one year is roughly 25,000 bars on a 24/5 CFD), |
//|  and read the journal for the output path.                        |
//|                                                                  |
//|  Columns: time,open,high,low,close,tick_volume,spread             |
//|  spread is in POINTS, exactly as the terminal recorded it, so the |
//|  Python replay can charge the real historical spread instead of a |
//|  guess.                                                           |
//+------------------------------------------------------------------+
#property copyright "NAS100ML"
#property link      "https://github.com/ParadiseDevelopment/Nasdaq"
#property version   "1.00"
#property script_show_inputs
#property description "Export chart bars to CSV for the Python backtest replica"

input int    InpBars     = 30000;                  // Bars to export (0 = all available)
input string InpFileName = "NAS100ML/bars.csv";    // Output path under MQL5\Files

//+------------------------------------------------------------------+
void OnStart()
  {
   int available = Bars(_Symbol, _Period);
   if(available <= 0)
     {
      Print("ExportBars: no history for ", _Symbol, " ", EnumToString(_Period),
            " - open the chart and scroll back to download it first");
      return;
     }

   int want = (InpBars <= 0) ? available : MathMin(InpBars, available);

   MqlRates rates[];
   ArraySetAsSeries(rates, false);          // oldest first, which is replay order
   int copied = CopyRates(_Symbol, _Period, 0, want, rates);
   if(copied <= 0)
     {
      Print("ExportBars: CopyRates failed, error ", GetLastError());
      return;
     }

   int fh = FileOpen(InpFileName, FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      Print("ExportBars: cannot open '", InpFileName, "', error ", GetLastError());
      return;
     }

   //--- a header line the replica uses to sanity-check the contract size
   //--- and tick value it should price trades with
   FileWriteString(fh, StringFormat("#symbol=%s timeframe=%s digits=%d point=%.10g "
                                    "tick_value=%.10g tick_size=%.10g "
                                    "volume_min=%.10g volume_step=%.10g volume_max=%.10g\r\n",
                                    _Symbol, EnumToString(_Period),
                                    (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS),
                                    SymbolInfoDouble(_Symbol, SYMBOL_POINT),
                                    SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE),
                                    SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE),
                                    SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN),
                                    SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP),
                                    SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX)));

   FileWriteString(fh, "time,open,high,low,close,tick_volume,spread\r\n");

   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   for(int i = 0; i < copied; i++)
     {
      FileWriteString(fh, StringFormat("%s,%.*f,%.*f,%.*f,%.*f,%I64d,%d\r\n",
                                       TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES),
                                       digits, rates[i].open,
                                       digits, rates[i].high,
                                       digits, rates[i].low,
                                       digits, rates[i].close,
                                       rates[i].tick_volume,
                                       (int)rates[i].spread));
     }

   FileClose(fh);

   PrintFormat("ExportBars: wrote %d bars of %s %s to MQL5\\Files\\%s  (%s -> %s)",
               copied, _Symbol, EnumToString(_Period), InpFileName,
               TimeToString(rates[0].time, TIME_DATE | TIME_MINUTES),
               TimeToString(rates[copied - 1].time, TIME_DATE | TIME_MINUTES));
  }
//+------------------------------------------------------------------+
