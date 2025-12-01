import re
import logging

logger = logging.getLogger(__name__)


class NormalizationService:
    """
    Service for normalizing vehicle descriptions to a standard format.
    
    This handles the differences between partner descriptions and our database format:
    - Partner: "RENAULT MEGANE 1.6 COMFORT MT 2009, 108CV, 108CV, SEDAN, SEDAN, COMBUSTION, COMBUSTION, MT, MT"
    - Database: "RENAULT MEGANE 2009 CONFORT L4 1.6L 183 CP 4 PUERTAS STD BA AA"
    """
    
    # Transmission synonyms -> standardized value
    TRANSMISSION_MAPPINGS = {
        r'\bMT\b': 'STD',
        r'\bM/T\b': 'STD',
        r'\bMANUAL\b': 'STD',
        r'\bESTANDAR\b': 'STD',
        r'\bESTÁNDAR\b': 'STD',
        r'\bAT\b': 'AUT',
        r'\bA/T\b': 'AUT',
        r'\bAUTO\b': 'AUT',
        r'\bAUTOMATICO\b': 'AUT',
        r'\bAUTOMÁTICO\b': 'AUT',
        r'\bAUTOMATICA\b': 'AUT',
        r'\bAUTOMÁTICA\b': 'AUT',
    }
    
    # Power unit synonyms -> standardized value
    POWER_MAPPINGS = {
        r'\bCV\b': 'CP',
        r'\bHP\b': 'CP',
        r'\bBHP\b': 'CP',
        r'\bCABALLOS\b': 'CP',
    }
    
    # Body type synonyms -> standardized value
    BODY_MAPPINGS = {
        r'\bSEDAN\b': 'SEDAN',
        r'\bSEDÁN\b': 'SEDAN',
        r'\b4\s*PUERTAS\b': '4 PUERTAS',
        r'\b4P\b': '4 PUERTAS',
        r'\b4DR\b': '4 PUERTAS',
        r'\b5\s*PUERTAS\b': '5 PUERTAS',
        r'\b5P\b': '5 PUERTAS',
        r'\b5DR\b': '5 PUERTAS',
        r'\b2\s*PUERTAS\b': '2 PUERTAS',
        r'\b2P\b': '2 PUERTAS',
        r'\b2DR\b': '2 PUERTAS',
        r'\b3\s*PUERTAS\b': '3 PUERTAS',
        r'\b3P\b': '3 PUERTAS',
        r'\b3DR\b': '3 PUERTAS',
        r'\bHATCHBACK\b': 'HB',
        r'\bHATCH\b': 'HB',
        r'\bPICKUP\b': 'PICKUP',
        r'\bPICK-UP\b': 'PICKUP',
        r'\bDOBLE\s*CABINA\b': 'DOBLE CABINA',
        r'\bD/C\b': 'DOBLE CABINA',
    }
    
    # Fuel type synonyms -> standardized value
    FUEL_MAPPINGS = {
        r'\bCOMBUSTION\b': 'GASOLINA',
        r'\bGASOLINA\b': 'GASOLINA',
        r'\bNAFTA\b': 'GASOLINA',
        r'\bBENCINA\b': 'GASOLINA',
        r'\bDIESEL\b': 'DIESEL',
        r'\bDIÉSEL\b': 'DIESEL',
        r'\bTDI\b': 'DIESEL',
        r'\bHDI\b': 'DIESEL',
        r'\bCDTI\b': 'DIESEL',
        r'\bDCI\b': 'DIESEL',
        r'\bHIBRIDO\b': 'HEV',
        r'\bHÍBRIDO\b': 'HEV',
        r'\bHYBRID\b': 'HEV',
        r'\bELECTRICO\b': 'EV',
        r'\bELÉCTRICO\b': 'EV',
        r'\bBEV\b': 'EV',
        r'\b100%\s*ELECTRICO\b': 'EV',
        r'\b100%\s*ELÉCTRICO\b': 'EV',
    }
    
    # Drive type synonyms -> standardized value
    DRIVE_MAPPINGS = {
        r'\b4WD\b': '4X4',
        r'\bAWD\b': '4X4',
        r'\b4X4\b': '4X4',
    }
    
    # Engine displacement patterns
    ENGINE_PATTERNS = [
        # 1600CC -> 1.6L
        (r'\b(\d{3,4})CC\b', lambda m: f"{int(m.group(1))/1000:.1f}L"),
        # 1,6 -> 1.6
        (r'\b(\d),(\d)\b', r'\1.\2'),
        # Ensure L suffix for displacement without it (e.g., "1.6" -> "1.6L" when followed by version/other info)
        (r'\b(\d\.\d)(?=\s+[A-Z])', r'\1L'),
    ]

    def normalize(self, description: str, full_normalization: bool = False) -> str:
        """
        Normalize a vehicle description to a standard format.
        
        Basic normalization (always applied):
        1. Converts to uppercase
        5. Cleans up spacing and punctuation (fixes double commas, extra spaces)
        
        Full normalization (only when full_normalization=True):
        2. Removes duplicate consecutive values
        3. Standardizes terminology (transmission, power, body, fuel, drive)
        4. Normalizes engine displacement format
        
        Args:
            description: The raw vehicle description from a partner
            full_normalization: If True, applies full normalization including 
                               terminology standardization and duplicate removal. 
                               Defaults to False.
            
        Returns:
            Normalized description string
        """
        if not description:
            return ""
        
        original = description
        
        # Step 1: Convert to uppercase (always applied)
        normalized = description.upper().strip()
        
        # Step 5 (partial): Fix double commas and extra commas (always applied)
        normalized = re.sub(r',\s*,', ',', normalized)
        normalized = re.sub(r',\s*$', '', normalized)
        normalized = re.sub(r'^\s*,', '', normalized)
        
        if full_normalization:
            # Step 2: Remove duplicate consecutive values (e.g., "SEDAN, SEDAN" -> "SEDAN")
            normalized = self._remove_duplicates(normalized)
            
            # Step 3: Apply all terminology mappings
            normalized = self._apply_mappings(normalized, self.TRANSMISSION_MAPPINGS)
            normalized = self._apply_mappings(normalized, self.POWER_MAPPINGS)
            normalized = self._apply_mappings(normalized, self.BODY_MAPPINGS)
            normalized = self._apply_mappings(normalized, self.FUEL_MAPPINGS)
            normalized = self._apply_mappings(normalized, self.DRIVE_MAPPINGS)
            
            # Step 4: Normalize engine displacement
            normalized = self._normalize_engine(normalized)
        
        # Step 5 (continued): Clean up whitespace (always applied)
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\s*,\s*', ', ', normalized)
        normalized = normalized.strip()
        
        logger.debug(
            f"Normalized description (full_normalization={full_normalization}): "
            f"'{original}' -> '{normalized}'"
        )
        
        return normalized
    
    def _remove_duplicates(self, text: str) -> str:
        """
        Remove duplicate consecutive values separated by commas.
        
        Example: "SEDAN, SEDAN, MT, MT" -> "SEDAN, MT"
        """
        # Split by comma
        parts = [p.strip() for p in text.split(',')]
        
        # Remove consecutive duplicates
        result = []
        for part in parts:
            if not result or part != result[-1]:
                result.append(part)
        
        return ', '.join(result)
    
    def _apply_mappings(self, text: str, mappings: dict) -> str:
        """Apply a set of regex mappings to standardize terminology."""
        for pattern, replacement in mappings.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _normalize_engine(self, text: str) -> str:
        """Normalize engine displacement formats."""
        for pattern, replacement in self.ENGINE_PATTERNS:
            if callable(replacement):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            else:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

