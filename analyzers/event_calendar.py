"""
Event Calendar Analyzer
Tracks major economic events (FOMC, CPI, NFP) to avoid trading during high-volatility periods.
All times are handled in UTC internally, with Paris timezone for display.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

# Try to import zoneinfo (Python 3.9+), fallback to pytz
try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from pytz import timezone as ZoneInfo
    except ImportError:
        ZoneInfo = None


class EventCalendarAnalyzer:
    """
    Tracks major economic events that can cause high volatility.
    
    During event windows (4h before → 2h after), signals receive a confidence penalty.
    This helps avoid being caught in unpredictable moves around announcements.
    
    All times stored as UTC datetime objects.
    User's timezone: Europe/Paris (CET/CEST)
    """
    
    # Paris timezone
    PARIS_TZ = ZoneInfo("Europe/Paris") if ZoneInfo else None
    
    # FOMC Meeting dates 2025-2026 (announcement typically at 19:00 UTC / 20:00 Paris winter)
    FOMC_DATES = [
        # 2025 - Times are approximate announcement times in UTC
        datetime(2025, 1, 29, 19, 0),
        datetime(2025, 3, 19, 18, 0),
        datetime(2025, 5, 7, 18, 0),
        datetime(2025, 6, 18, 18, 0),
        datetime(2025, 7, 30, 18, 0),
        datetime(2025, 9, 17, 18, 0),
        datetime(2025, 11, 5, 19, 0),
        datetime(2025, 12, 17, 19, 0),
        # 2026
        datetime(2026, 1, 28, 19, 0),
        datetime(2026, 3, 18, 18, 0),
        datetime(2026, 5, 6, 18, 0),
        datetime(2026, 6, 17, 18, 0),
        datetime(2026, 7, 29, 18, 0),
        datetime(2026, 9, 16, 18, 0),
        datetime(2026, 11, 4, 19, 0),
        datetime(2026, 12, 16, 19, 0),
    ]
    
    # CPI Release dates 2025-2026 (typically at 13:30 UTC / 14:30 Paris)
    # Usually around the 12th-13th of each month
    CPI_DATES = [
        # 2025
        datetime(2025, 1, 15, 13, 30),
        datetime(2025, 2, 12, 13, 30),
        datetime(2025, 3, 12, 13, 30),
        datetime(2025, 4, 10, 12, 30),  # Summer time
        datetime(2025, 5, 13, 12, 30),
        datetime(2025, 6, 11, 12, 30),
        datetime(2025, 7, 11, 12, 30),
        datetime(2025, 8, 13, 12, 30),
        datetime(2025, 9, 11, 12, 30),
        datetime(2025, 10, 10, 12, 30),
        datetime(2025, 11, 13, 13, 30),  # Winter time
        datetime(2025, 12, 11, 13, 30),
        # 2026
        datetime(2026, 1, 14, 13, 30),
        datetime(2026, 2, 11, 13, 30),
        datetime(2026, 3, 11, 13, 30),
        datetime(2026, 4, 14, 12, 30),
        datetime(2026, 5, 13, 12, 30),
        datetime(2026, 6, 10, 12, 30),
        datetime(2026, 7, 15, 12, 30),
        datetime(2026, 8, 12, 12, 30),
        datetime(2026, 9, 16, 12, 30),
        datetime(2026, 10, 14, 12, 30),
        datetime(2026, 11, 12, 13, 30),
        datetime(2026, 12, 10, 13, 30),
    ]
    
    # NFP (Non-Farm Payrolls) - First Friday of each month at 13:30 UTC
    # Generating dynamically would be complex, adding key 2025-2026 dates
    NFP_DATES = [
        # 2025
        datetime(2025, 1, 10, 13, 30),
        datetime(2025, 2, 7, 13, 30),
        datetime(2025, 3, 7, 13, 30),
        datetime(2025, 4, 4, 12, 30),
        datetime(2025, 5, 2, 12, 30),
        datetime(2025, 6, 6, 12, 30),
        datetime(2025, 7, 3, 12, 30),
        datetime(2025, 8, 1, 12, 30),
        datetime(2025, 9, 5, 12, 30),
        datetime(2025, 10, 3, 12, 30),
        datetime(2025, 11, 7, 13, 30),
        datetime(2025, 12, 5, 13, 30),
        # 2026
        datetime(2026, 1, 9, 13, 30),
        datetime(2026, 2, 6, 13, 30),
        datetime(2026, 3, 6, 13, 30),
        datetime(2026, 4, 3, 12, 30),
        datetime(2026, 5, 1, 12, 30),
        datetime(2026, 6, 5, 12, 30),
        datetime(2026, 7, 2, 12, 30),
        datetime(2026, 8, 7, 12, 30),
        datetime(2026, 9, 4, 12, 30),
        datetime(2026, 10, 2, 12, 30),
        datetime(2026, 11, 6, 13, 30),
        datetime(2026, 12, 4, 13, 30),
    ]
    
    # Event windows and penalties
    WINDOW_BEFORE = timedelta(hours=4)  # 4 hours before event
    WINDOW_AFTER = timedelta(hours=2)   # 2 hours after event
    
    IMPACT_LEVELS = {
        'FOMC': {'level': 'EXTREME', 'penalty': 40},
        'CPI': {'level': 'HIGH', 'penalty': 30},
        'NFP': {'level': 'HIGH', 'penalty': 30},
    }
    
    def __init__(self):
        self.all_events = self._build_event_list()
        
    def _build_event_list(self) -> List[Dict]:
        """Build a sorted list of all events"""
        events = []
        
        for dt in self.FOMC_DATES:
            events.append({
                'datetime': dt,
                'type': 'FOMC',
                'name': 'FOMC Meeting',
                'impact': self.IMPACT_LEVELS['FOMC']
            })
            
        for dt in self.CPI_DATES:
            events.append({
                'datetime': dt,
                'type': 'CPI',
                'name': 'CPI Release',
                'impact': self.IMPACT_LEVELS['CPI']
            })
            
        for dt in self.NFP_DATES:
            events.append({
                'datetime': dt,
                'type': 'NFP',
                'name': 'NFP Jobs Report',
                'impact': self.IMPACT_LEVELS['NFP']
            })
            
        # Sort by datetime
        events.sort(key=lambda x: x['datetime'])
        return events
    
    def _get_current_utc(self) -> datetime:
        """Get current time in UTC"""
        return datetime.utcnow()
    
    def _format_time_to_event(self, delta: timedelta) -> str:
        """Format time delta to human readable string"""
        total_seconds = delta.total_seconds()
        hours = int(abs(total_seconds) // 3600)
        minutes = int((abs(total_seconds) % 3600) // 60)
        
        if total_seconds < 0:
            return f"-{hours}h{minutes:02d}m"  # After event
        else:
            return f"+{hours}h{minutes:02d}m"  # Before event
    
    def _to_paris_time(self, dt: datetime) -> str:
        """Convert UTC datetime to Paris time string"""
        if self.PARIS_TZ:
            try:
                # Make datetime timezone-aware if it isn't
                if dt.tzinfo is None:
                    from datetime import timezone
                    dt = dt.replace(tzinfo=timezone.utc)
                paris_dt = dt.astimezone(self.PARIS_TZ)
                return paris_dt.strftime("%d/%m %H:%M")
            except Exception:
                pass
        return dt.strftime("%d/%m %H:%M UTC")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze current event status
        Returns whether we're in an event window and relevant details
        """
        now = self._get_current_utc()
        
        # Default response
        result = {
            'event_active': False,
            'current_event': None,
            'event_type': None,
            'event_time_paris': None,
            'time_to_event': None,
            'impact_level': None,
            'confidence_penalty': 0,
            'next_event': None,
            'next_event_time_paris': None,
            'next_event_in': None
        }
        
        # Check each event
        for event in self.all_events:
            event_dt = event['datetime']
            
            # Calculate window
            window_start = event_dt - self.WINDOW_BEFORE
            window_end = event_dt + self.WINDOW_AFTER
            
            # Check if we're in the window
            if window_start <= now <= window_end:
                time_delta = event_dt - now
                result.update({
                    'event_active': True,
                    'current_event': event['name'],
                    'event_type': event['type'],
                    'event_time_paris': self._to_paris_time(event_dt),
                    'time_to_event': self._format_time_to_event(time_delta),
                    'impact_level': event['impact']['level'],
                    'confidence_penalty': event['impact']['penalty']
                })
                break
            
            # Find next upcoming event
            if event_dt > now and result['next_event'] is None:
                time_to_next = event_dt - now
                hours_to_next = time_to_next.total_seconds() / 3600
                
                result.update({
                    'next_event': event['name'],
                    'next_event_time_paris': self._to_paris_time(event_dt),
                    'next_event_in': f"{hours_to_next:.1f}h"
                })
        
        return result
