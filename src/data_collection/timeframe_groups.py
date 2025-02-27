from enum import Enum
from typing import List, Dict

class TimeframeGroup(Enum):
    SHORT_MEDIUM = "short_medium"
    LONG = "long"

class TimeframeManager:
    def __init__(self):
        self.timeframe_groups = {
            TimeframeGroup.SHORT_MEDIUM: ['5m', '15m', '30m', '1h', '4h', '1d'],
            TimeframeGroup.LONG: ['1w', '1M']
        }
        
    def get_timeframes(self, group: TimeframeGroup) -> List[str]:
        return self.timeframe_groups[group]
    
    def get_base_timeframe(self, group: TimeframeGroup) -> str:
        """Retourne le timeframe de référence pour l'alignement dans chaque groupe"""
        if group == TimeframeGroup.SHORT_MEDIUM:
            return '1d'
        return '1M' 