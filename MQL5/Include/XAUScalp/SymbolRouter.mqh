//+------------------------------------------------------------------+
//|                                                 SymbolRouter.mqh |
//|  Decides which gold symbol is tradable right now.                 |
//|                                                                  |
//|  Weekdays  -> the spot symbol   (XAUUSD.s)                        |
//|  Weekends  -> the 24/7 symbol   (XAUUSD24/7.s)                    |
//|                                                                  |
//|  The decision is taken from the broker's own trading session      |
//|  table, not from a hard-coded clock, so it survives DST shifts    |
//|  and holiday closes without being re-tuned. The nightly           |
//|  maintenance break is deliberately NOT treated as a weekend: a    |
//|  closed stretch only counts as one once it exceeds                |
//|  WeekendGapHours (default 6h).                                    |
//+------------------------------------------------------------------+
#ifndef __XAUSCALP_SYMBOLROUTER_MQH__
#define __XAUSCALP_SYMBOLROUTER_MQH__

#define XS_WEEK_SECONDS   604800
#define XS_MAX_INTERVALS  64

//--- which symbol the EA should be working on
enum ENUM_XS_ACTIVE
  {
   XS_ACTIVE_NONE    = 0,   // both markets shut - stand down
   XS_ACTIVE_WEEKDAY = 1,   // the spot symbol
   XS_ACTIVE_WEEKEND = 2    // the 24/7 symbol
  };

//--- how the weekend is detected
enum ENUM_XS_ROUTE
  {
   XS_ROUTE_AUTO         = 0, // broker session table (recommended)
   XS_ROUTE_CLOCK        = 1, // fixed Fri-close / Mon-open server clock
   XS_ROUTE_WEEKDAY_ONLY = 2  // never trade the weekend symbol
  };

//+------------------------------------------------------------------+
//| A symbol's open/closed map for one week, in "week seconds"        |
//| (0 = Sunday 00:00:00 server time, 604800 = the next Sunday).      |
//+------------------------------------------------------------------+
class CWeekCalendar
  {
private:
   int               m_from[XS_MAX_INTERVALS];
   int               m_to[XS_MAX_INTERVALS];
   int               m_count;
   bool              m_valid;

   //--- add one interval, splitting it if it runs past the week end
   void              Add(const int f, const int t)
     {
      if(t <= f)
         return;
      if(t > XS_WEEK_SECONDS)
        {
         Add(f, XS_WEEK_SECONDS);
         Add(0, t - XS_WEEK_SECONDS);
         return;
        }
      if(m_count >= XS_MAX_INTERVALS)
         return;
      m_from[m_count] = f;
      m_to[m_count]   = t;
      m_count++;
     }

public:
                     CWeekCalendar(void) : m_count(0), m_valid(false) {}

   void              Clear(void) { m_count = 0; m_valid = false; }

   //--- explicit copy: MQL5 does not promise a member-wise one for a class
   //--- holding static arrays
   void              CopyFrom(const CWeekCalendar &src)
     {
      m_count = src.Count();
      m_valid = src.IsValid();
      for(int i = 0; i < m_count; i++)
        {
         m_from[i] = src.From(i);
         m_to[i]   = src.To(i);
        }
     }

   int               From(const int i) const { return(i >= 0 && i < m_count ? m_from[i] : 0); }
   int               To(const int i)   const { return(i >= 0 && i < m_count ? m_to[i]   : 0); }
   bool              IsValid(void) const { return(m_valid && m_count > 0); }
   int               Count(void)   const { return(m_count); }

   //+---------------------------------------------------------------+
   //| Read the broker's quoting/trading sessions for the symbol.     |
   //+---------------------------------------------------------------+
   bool              BuildFromSessions(const string symbol)
     {
      Clear();
      datetime from, to;
      for(int d = 0; d < 7; d++)
        {
         for(int s = 0; s < 8; s++)
           {
            if(!SymbolInfoSessionTrade(symbol, (ENUM_DAY_OF_WEEK)d, s, from, to))
               break;
            int f = d * 86400 + (int)from;
            int t = d * 86400 + (int)to;
            //--- a session ending at midnight is reported as 00:00
            if(t <= f)
              {
               if((int)to == 0)
                  t = f + (86400 - (int)from);
               else
                  continue;
              }
            Add(f, t);
           }
        }
      m_valid = (m_count > 0);
      return(m_valid);
     }

   //+---------------------------------------------------------------+
   //| A single continuous window, e.g. Monday 01:00 -> Friday 23:45. |
   //| dow: 0 = Sunday .. 6 = Saturday.                               |
   //+---------------------------------------------------------------+
   bool              BuildFromWindow(const int startDow, const int startSec,
                                     const int endDow,   const int endSec)
     {
      Clear();
      int f = startDow * 86400 + startSec;
      int t = endDow   * 86400 + endSec;
      if(t <= f)
         t += XS_WEEK_SECONDS;      // window wraps the week boundary
      Add(f, t);
      m_valid = (m_count > 0);
      return(m_valid);
     }

   //+---------------------------------------------------------------+
   //| Is the market open at this point of the week?                  |
   //+---------------------------------------------------------------+
   bool              IsOpen(const int ws) const
     {
      for(int i = 0; i < m_count; i++)
         if(ws >= m_from[i] && ws < m_to[i])
            return(true);
      return(false);
     }

   //+---------------------------------------------------------------+
   //| Seconds until the market shuts. Contiguous sessions (a daily   |
   //| table that runs 00:00-24:00 on consecutive days) are walked    |
   //| through, so this returns the end of the whole open stretch.    |
   //| -1 when the market is already closed.                          |
   //+---------------------------------------------------------------+
   int               SecondsToClose(const int ws) const
     {
      int idx = -1;
      for(int i = 0; i < m_count; i++)
         if(ws >= m_from[i] && ws < m_to[i])
           {
            idx = i;
            break;
           }
      if(idx < 0)
         return(-1);

      int edge = m_to[idx];
      for(int guard = 0; guard < XS_MAX_INTERVALS; guard++)
        {
         int probe = edge % XS_WEEK_SECONDS;
         int next  = -1;
         for(int i = 0; i < m_count; i++)
            if(m_from[i] == probe)
              {
               next = i;
               break;
              }
         if(next < 0)
            break;
         edge += (m_to[next] - m_from[next]);
        }
      int delta = edge - ws;
      if(delta < 0)
         delta += XS_WEEK_SECONDS;
      //--- a symbol whose table covers the whole week never closes
      if(delta > XS_WEEK_SECONDS)
         delta = XS_WEEK_SECONDS;
      return(delta);
     }

   //+---------------------------------------------------------------+
   //| Seconds until the market opens. 0 when it already is.          |
   //+---------------------------------------------------------------+
   int               SecondsToOpen(const int ws) const
     {
      if(IsOpen(ws))
         return(0);
      int best = XS_WEEK_SECONDS;
      for(int i = 0; i < m_count; i++)
        {
         int d = m_from[i] - ws;
         while(d < 0)
            d += XS_WEEK_SECONDS;
         //--- an interval that merely continues another one is not an open
         bool contiguous = false;
         for(int j = 0; j < m_count; j++)
            if(m_to[j] == m_from[i])
               contiguous = true;
         if(contiguous)
            continue;
         if(d < best)
            best = d;
        }
      return(best);
     }

   //+---------------------------------------------------------------+
   //| Seconds since the market shut. 0 when it is open.              |
   //+---------------------------------------------------------------+
   int               SecondsSinceClose(const int ws) const
     {
      if(IsOpen(ws))
         return(0);
      int best = XS_WEEK_SECONDS;
      for(int i = 0; i < m_count; i++)
        {
         int d = ws - m_to[i];
         while(d < 0)
            d += XS_WEEK_SECONDS;
         bool contiguous = false;
         for(int j = 0; j < m_count; j++)
            if(m_from[j] == m_to[i])
               contiguous = true;
         if(contiguous)
            continue;
         if(d < best)
            best = d;
        }
      return(best);
     }

   //+---------------------------------------------------------------+
   //| Total length of the closed stretch we are sitting in. This is  |
   //| what separates a one-hour maintenance break from a weekend.    |
   //+---------------------------------------------------------------+
   int               CurrentGapSeconds(const int ws) const
     {
      if(IsOpen(ws))
         return(0);
      return(SecondsSinceClose(ws) + SecondsToOpen(ws));
     }
  };

//+------------------------------------------------------------------+
//| Picks the live symbol and reports how long it stays live.         |
//+------------------------------------------------------------------+
class CSymbolRouter
  {
private:
   string            m_weekday;
   string            m_weekend;
   ENUM_XS_ROUTE     m_mode;
   int               m_weekendGapSec;
   CWeekCalendar     m_calWeekday;
   bool              m_weekendUsable;
   datetime          m_lastBuild;
   string            m_note;

   //--- seconds into the trading week, 0 = Sunday 00:00 server time
   static int        WeekSeconds(const datetime t)
     {
      MqlDateTime d;
      TimeToStruct(t, d);
      return(d.day_of_week * 86400 + d.hour * 3600 + d.min * 60 + d.sec);
     }

   //--- uppercase, with the punctuation brokers disagree about removed
   static string     Normalise(const string s)
     {
      string out = "";
      int n = StringLen(s);
      for(int i = 0; i < n; i++)
        {
         ushort c = StringGetCharacter(s, i);
         if(c == '.' || c == '/' || c == '_' || c == '-' || c == ' ')
            continue;
         if(c >= 'a' && c <= 'z')
            c -= 32;
         out += ShortToString(c);
        }
      return(out);
     }

public:
                     CSymbolRouter(void) : m_weekday(""), m_weekend(""),
                                           m_mode(XS_ROUTE_AUTO), m_weekendGapSec(21600),
                                           m_weekendUsable(false), m_lastBuild(0), m_note("") {}

   string            WeekdaySymbol(void) const { return(m_weekday); }
   string            WeekendSymbol(void) const { return(m_weekend); }
   bool              WeekendUsable(void) const { return(m_weekendUsable); }
   string            Note(void)          const { return(m_note); }

   //+---------------------------------------------------------------+
   //| Find a symbol by name, tolerating the punctuation and case a   |
   //| broker may have chosen ("XAUUSD24/7.s" vs "XAUUSD247.s").      |
   //| Returns "" when nothing matches. Selects it in Market Watch.   |
   //+---------------------------------------------------------------+
   static string     ResolveSymbol(const string wanted)
     {
      if(wanted == "")
         return("");
      if(SymbolSelect(wanted, true))
         return(wanted);

      string want  = Normalise(wanted);
      int    total = SymbolsTotal(false);

      //--- exact match once punctuation and case are stripped
      for(int i = 0; i < total; i++)
        {
         string name = SymbolName(i, false);
         if(Normalise(name) == want)
           {
            SymbolSelect(name, true);
            return(name);
           }
        }
      //--- the broker may add a suffix we do not know about
      for(int i = 0; i < total; i++)
        {
         string name = SymbolName(i, false);
         if(StringFind(Normalise(name), want) >= 0)
           {
            SymbolSelect(name, true);
            return(name);
           }
        }
      //--- or we may be asking with a suffix the broker does not use. Only
      //--- accept that when what we drop carries no digits: otherwise
      //--- "XAUUSD24/7.s" happily resolves to plain "XAUUSD", which is the
      //--- one mistake this whole class exists to avoid.
      for(int i = 0; i < total; i++)
        {
         string name = SymbolName(i, false);
         string norm = Normalise(name);
         int    at   = StringFind(want, norm);
         if(at != 0 || StringLen(norm) < 4)
            continue;
         string extra = StringSubstr(want, StringLen(norm));
         bool   hasDigit = false;
         for(int k = 0; k < StringLen(extra); k++)
           {
            ushort ch = StringGetCharacter(extra, k);
            if(ch >= '0' && ch <= '9')
               hasDigit = true;
           }
         if(hasDigit)
            continue;
         SymbolSelect(name, true);
         return(name);
        }
      return("");
     }

   //+---------------------------------------------------------------+
   //| weekendGapHours: a closed stretch shorter than this is a       |
   //| maintenance break, not a weekend.                              |
   //+---------------------------------------------------------------+
   bool              Init(const string weekdaySymbol, const string weekendSymbol,
                          const ENUM_XS_ROUTE mode, const double weekendGapHours,
                          const int weekStartDow, const int weekStartSec,
                          const int weekEndDow,   const int weekEndSec)
     {
      m_mode          = mode;
      m_weekendGapSec = (int)MathMax(3600.0, weekendGapHours * 3600.0);

      m_weekday = ResolveSymbol(weekdaySymbol);
      if(m_weekday == "")
        {
         m_note = "weekday symbol '" + weekdaySymbol + "' not found at this broker";
         return(false);
        }

      m_weekend = (mode == XS_ROUTE_WEEKDAY_ONLY) ? "" : ResolveSymbol(weekendSymbol);
      m_weekendUsable = (m_weekend != "") &&
                        ((ENUM_SYMBOL_TRADE_MODE)SymbolInfoInteger(m_weekend, SYMBOL_TRADE_MODE)
                         == SYMBOL_TRADE_MODE_FULL);

      if(mode == XS_ROUTE_CLOCK)
         m_calWeekday.BuildFromWindow(weekStartDow, weekStartSec, weekEndDow, weekEndSec);
      else if(!m_calWeekday.BuildFromSessions(m_weekday))
        {
         //--- no session table (some tester configurations): fall back to
         //--- the configured clock rather than trading blind
         m_calWeekday.BuildFromWindow(weekStartDow, weekStartSec, weekEndDow, weekEndSec);
         m_note = "no session table for " + m_weekday + " - using the configured clock window";
        }

      m_lastBuild = TimeCurrent();
      return(m_calWeekday.IsValid());
     }

   //+---------------------------------------------------------------+
   //| Re-read the session table occasionally: brokers publish DST    |
   //| and holiday changes through it.                                |
   //+---------------------------------------------------------------+
   void              Refresh(void)
     {
      if(m_mode == XS_ROUTE_CLOCK)
         return;
      datetime now = TimeCurrent();
      if(now - m_lastBuild < 3600)
         return;
      m_lastBuild = now;
      CWeekCalendar fresh;
      if(fresh.BuildFromSessions(m_weekday))
         m_calWeekday.CopyFrom(fresh);
     }

   //+---------------------------------------------------------------+
   //| The routing decision.                                          |
   //+---------------------------------------------------------------+
   ENUM_XS_ACTIVE    Resolve(void)
     {
      int ws = WeekSeconds(TimeCurrent());

      if(m_calWeekday.IsOpen(ws))
         return(XS_ACTIVE_WEEKDAY);

      if(m_mode == XS_ROUTE_WEEKDAY_ONLY || !m_weekendUsable)
         return(XS_ACTIVE_NONE);

      //--- closed. Weekend, or just the nightly break?
      if(m_calWeekday.CurrentGapSeconds(ws) < m_weekendGapSec)
         return(XS_ACTIVE_NONE);

      return(XS_ACTIVE_WEEKEND);
     }

   string            SymbolFor(const ENUM_XS_ACTIVE a) const
     {
      if(a == XS_ACTIVE_WEEKDAY) return(m_weekday);
      if(a == XS_ACTIVE_WEEKEND) return(m_weekend);
      return("");
     }

   //+---------------------------------------------------------------+
   //| Seconds until the active symbol must be handed over:           |
   //|   weekday active -> the spot market closes                     |
   //|   weekend active -> the spot market re-opens                   |
   //| Anything above a week is reported as a week.                   |
   //+---------------------------------------------------------------+
   int               SecondsToHandover(const ENUM_XS_ACTIVE a)
     {
      int ws = WeekSeconds(TimeCurrent());
      if(a == XS_ACTIVE_WEEKDAY)
        {
         int s = m_calWeekday.SecondsToClose(ws);
         return(s < 0 ? 0 : s);
        }
      if(a == XS_ACTIVE_WEEKEND)
         return(m_calWeekday.SecondsToOpen(ws));
      return(XS_WEEK_SECONDS);
     }

   static string     FormatSpan(const int seconds)
     {
      int s = (int)MathMax(0, seconds);
      int h = s / 3600;
      int m = (s % 3600) / 60;
      if(h >= 24)
         return(StringFormat("%dd %02dh", h / 24, h % 24));
      return(StringFormat("%02dh %02dm", h, m));
     }

   //--- human-readable state for the chart panel
   string            Describe(void)
     {
      int ws = WeekSeconds(TimeCurrent());
      if(m_calWeekday.IsOpen(ws))
         return(StringFormat("%s open, closes in %s",
                             m_weekday, FormatSpan(m_calWeekday.SecondsToClose(ws))));
      return(StringFormat("%s shut for %s, opens in %s",
                          m_weekday,
                          FormatSpan(m_calWeekday.SecondsSinceClose(ws)),
                          FormatSpan(m_calWeekday.SecondsToOpen(ws))));
     }

  };

#endif // __XAUSCALP_SYMBOLROUTER_MQH__
//+------------------------------------------------------------------+
