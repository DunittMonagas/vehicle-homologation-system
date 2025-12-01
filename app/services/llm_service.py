import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from app.core.config import config
from app.core.constants import VALID_GEMINI_MODELS


logger = logging.getLogger(__name__)


VEHICLE_MATCHING_SYSTEM_PROMPT = """Eres un experto profesional automotriz con más de 20 años de experiencia en identificación, clasificación y homologación de vehículos. Trabajas para un sistema de homologación vehicular que recibe descripciones de vehículos de múltiples socios (partners), cada uno con su propia base de datos y convenciones de nomenclatura diferentes.

## CONTEXTO DEL PROBLEMA

Diferentes socios envían descripciones del mismo vehículo de formas muy distintas. Por ejemplo, el mismo vehículo puede ser descrito como:

- Partner A: "RENAULT MEGANE 1.6 COMFORT MT 2009, 108CV, 108CV, SEDAN, SEDAN, COMBUSTION, COMBUSTION, MT, MT"
- Partner B: "Renault Megane Comfort Manual 1.6L 2009 Sedán"
- Partner C: "MEGANE COMFORT 2009 1600CC MT 4 PUERTAS"

Todos se refieren al mismo vehículo en nuestro catálogo:
"Renault Megane Comfort 1.6 MT 2009, SEDAN, 108HP, COMBUSTION"

## DESAFÍOS TÉCNICOS QUE DEBES MANEJAR

1. **Variaciones de formato**: Órdenes diferentes, abreviaturas, términos en español/inglés
2. **Información redundante**: Datos repetidos o campos concatenados (ej: "SEDAN, SEDAN", "MT, MT")
3. **Información faltante**: Algunos partners omiten campos como marca, tipo de combustible, etc.
4. **Sinónimos y equivalencias**:
   - Transmisión: "MT" = "Manual" = "M/T" = "STD" = "Estándar"
   - Transmisión: "AT" = "Automático" = "A/T" = "Automatica" = "CVT" (en algunos contextos)
   - Carrocería: "Sedán" = "SEDAN" = "4 PUERTAS" = "4P" = "4DR"
   - Carrocería: "Hatchback" = "HB" = "5 PUERTAS" = "5P"
   - Carrocería: "SUV" = "Camioneta" = "4x4" (en algunos contextos)
   - Carrocería: "Pickup" = "Pick-up" = "Cabina" = "Doble Cabina" = "D/C"
   - Potencia: "CV" = "HP" = "Caballos" = "BHP"
   - Motor: "1.6" = "1.6L" = "1600CC" = "1600" = "1,6"
   - Combustible: "COMBUSTION" = "Gasolina" = "Nafta" = "Bencina"
   - Combustible: "DIESEL" = "Diésel" = "TDI" = "HDI" = "CDTI" = "DCI"
   - Combustible: "Híbrido" = "HEV" = "HYBRID"
   - Combustible: "Eléctrico" = "EV" = "BEV" = "100% Eléctrico"
5. **Ambigüedad**: Múltiples versiones del mismo modelo con especificaciones similares

## CONOCIMIENTO ESPECÍFICO

- Conoces todas las marcas principales: Toyota, Nissan, Honda, Mazda, Chevrolet, Ford, Volkswagen, Renault, Peugeot, Kia, Hyundai, BMW, Mercedes-Benz, Audi, SEAT, Fiat, Jeep, Dodge, RAM, Mitsubishi, Suzuki, Subaru, etc.
- Conoces las variantes regionales de nombres (ej: Chevrolet Aveo = Pontiac G3 = Daewoo Kalos)
- Entiendes nomenclaturas de versiones: LE, SE, XLE, Limited, Sport, Comfort, Luxury, Base, etc.
- Reconoces que el año modelo puede diferir del año de fabricación
- Comprendes los diferentes sistemas de medición de motor (CC, litros)

## EJEMPLOS DE MATCHING

Ejemplo 1 - Información redundante:
- Input: "TOYOTA COROLLA 1.8 LE 2020, SEDAN, SEDAN, AT, AT, GASOLINA"
- Match: "Toyota Corolla LE 1.8 AT 2020, SEDAN, GASOLINA" ✓

Ejemplo 2 - Formato diferente con sinónimos:
- Input: "Corolla 2020 automatico 1800cc sedán"
- Match: "Toyota Corolla LE 1.8 AT 2020, SEDAN, GASOLINA" ✓

Ejemplo 3 - Información parcial:
- Input: "NISSAN VERSA 2019 MANUAL"
- Opciones: ["Nissan Versa Sense 1.6 MT 2019", "Nissan Versa Advance 1.6 MT 2019"]
- Resultado: null (ambiguo entre versiones) ✗

Ejemplo 4 - Marca omitida pero modelo único:
- Input: "MUSTANG GT 5.0 2021 COUPE"
- Match: "Ford Mustang GT 5.0 V8 2021, COUPE, GASOLINA" ✓

Ejemplo 5 - Términos coloquiales:
- Input: "Tsuru 2017 4 puertas estándar"
- Match: "Nissan Tsuru GS I 1.6 MT 2017, SEDAN, GASOLINA" ✓

## REGLAS DE DECISIÓN

1. **Prioridad de campos para matching** (de mayor a menor importancia):
   - Marca + Modelo (juntos son críticos)
   - Año
   - Versión/Trim
   - Motor (cilindrada)
   - Transmisión
   - Tipo de carrocería
   - Combustible

2. **Cuándo seleccionar un vehículo**:
   - Los campos críticos (marca, modelo, año) coinciden o son claramente inferibles
   - Las diferencias son solo de formato o sinónimos
   - No hay ambigüedad significativa entre opciones

3. **Cuándo retornar null**:
   - Hay dos o más opciones que podrían ser correctas (ej: diferentes versiones del mismo modelo/año)
   - Falta información crítica que impide distinguir entre opciones
   - La descripción es demasiado vaga o genérica
   - Hay contradicciones evidentes entre la descripción y las opciones

4. **Nivel de confianza**:
   - 0.9-1.0: Match exacto o casi exacto, solo diferencias de formato
   - 0.7-0.9: Match con inferencias razonables
   - 0.5-0.7: Match probable pero con incertidumbre
   - <0.5: Deberías retornar null

## INSTRUCCIONES FINALES

Tu tarea es analizar la descripción del usuario y determinar cuál de las opciones disponibles corresponde al vehículo descrito. Debes:

1. Normalizar mentalmente ambas descripciones (usuario y opciones)
2. Identificar los campos clave presentes
3. Comparar usando equivalencias y sinónimos
4. Determinar si hay un match claro o si existe ambigüedad
5. Retornar el ID de la opción correcta o null si no puedes determinar con confianza

RECUERDA: Es preferible retornar null que hacer un match incorrecto. La precisión es más importante que el recall en este sistema."""


class VehicleOption(BaseModel):
    """A vehicle option from the database."""
    id: str = Field(description="The unique identifier of the vehicle in the database")
    description: str = Field(description="The description of the vehicle")


class VehicleMatchingRequest(BaseModel):
    """Request schema for vehicle matching."""
    user_description: str = Field(description="The user's description of the vehicle they are looking for")
    options: list[VehicleOption] = Field(description="List of vehicle options to choose from")


class VehicleMatchingResponse(BaseModel):
    """Structured response from the LLM for vehicle matching."""
    selected_id: Optional[str] = Field(
        default=None,
        description="The ID of the vehicle that best matches the user's description. Null if no confident match."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 for the match"
    )
    reasoning: str = Field(
        description="Explanation of why this vehicle was selected or why no match was found"
    )


class LLMService:
    """Service for interacting with Google's Gemini LLM."""
    
    def __init__(self):
        self.model_name = config.gemini_model
        self.api_key = config.gemini_api_key
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        if self.model_name not in VALID_GEMINI_MODELS:
            raise ValueError(
                f"Invalid Gemini model: {self.model_name}. "
                f"Valid options are: {', '.join(VALID_GEMINI_MODELS)}"
            )
        
        self.llm = self._initialize_llm()
        self.structured_llm = self._initialize_structured_llm()
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the base Gemini LLM with thinking mode enabled."""
        try:
            logger.info(f"Initializing Gemini LLM with model: {self.model_name}")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1,
                thinking_budget=1024,
                include_thoughts=True,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            raise
    
    def _initialize_structured_llm(self):
        """Initialize the LLM with structured output for vehicle matching."""
        return self.llm.with_structured_output(VehicleMatchingResponse)
    
    def match_vehicle(
        self,
        user_description: str,
        options: list[VehicleOption]
    ) -> VehicleMatchingResponse:
        """
        Use the LLM to determine which vehicle option best matches the user's description.
        
        Args:
            user_description: The user's description of the vehicle they are looking for
            options: List of vehicle options to choose from (from vector search results)
        
        Returns:
            VehicleMatchingResponse with selected_id (or None), confidence, and reasoning
        """
        if not options:
            logger.warning("No options provided for vehicle matching")
            return VehicleMatchingResponse(
                selected_id=None,
                confidence=0.0,
                reasoning="No vehicle options were provided to match against."
            )
        
        options_text = "\n".join([
            f"- ID: {opt.id} → {opt.description}"
            for opt in options
        ])
        
        user_message = f"""## DESCRIPCIÓN DEL VEHÍCULO ENVIADA POR EL PARTNER

"{user_description}"

## OPCIONES DISPONIBLES EN NUESTRO CATÁLOGO

{options_text}

## TU TAREA

Analiza la descripción del partner y determina cuál de las opciones de nuestro catálogo corresponde al mismo vehículo. Considera las variaciones de formato, sinónimos y campos redundantes o faltantes.

Retorna el ID de la opción que mejor coincida, o null si no puedes determinar con confianza cuál es el vehículo correcto (especialmente si hay ambigüedad entre versiones similares)."""
        
        try:
            logger.info(f"Matching vehicle description against {len(options)} options")
            
            messages = [
                SystemMessage(content=VEHICLE_MATCHING_SYSTEM_PROMPT),
                HumanMessage(content=user_message)
            ]
            
            response = self.structured_llm.invoke(messages)
            
            logger.info(
                f"Vehicle matching result: selected_id={response.selected_id}, "
                f"confidence={response.confidence:.2f}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error during vehicle matching: {e}")
            raise
    
    def match_vehicle_from_dict(
        self,
        user_description: str,
        options: list[dict]
    ) -> VehicleMatchingResponse:
        """
        Convenience method that accepts options as dictionaries.
        
        Args:
            user_description: The user's description of the vehicle
            options: List of dicts with 'id' and 'description' keys
        
        Returns:
            VehicleMatchingResponse with selected_id (or None), confidence, and reasoning
        """
        vehicle_options = [
            VehicleOption(id=opt["id"], description=opt["description"])
            for opt in options
        ]
        return self.match_vehicle(user_description, vehicle_options)

