import json
import logging

from app.core.config import config
from app.models.vehicle import Vehicle
from app.repositories.vehicle_repository import VehicleRepository
from app.services.llm_service import LLMService, VehicleOption
from app.services.normalization_service import NormalizationService
from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class VehicleService:
    def __init__(
        self,
        vehicle_repository: VehicleRepository,
        vector_service: VectorService,
        normalization_service: NormalizationService,
        llm_service: LLMService
    ):
        self.vector_service = vector_service
        self.vehicle_repository = vehicle_repository
        self.normalization_service = normalization_service
        self.llm_service = llm_service
        self.similarity_threshold = config.vector_similarity_threshold
        self.similarity_threshold_best_effort = config.vector_similarity_threshold_best_effort

    def get_vehicle_by_crabi_id(self, crabi_id: str) -> Vehicle | None:
        return self.vehicle_repository.get_by_crabi_id(crabi_id)

    def create_vehicle(self, id_crabi: str, description: str) -> Vehicle:
        vehicle = Vehicle(id_crabi=id_crabi, description=description)
        return self.vehicle_repository.create_vehicle(vehicle)

    def get_similar_vehicles(self, description: str, strict: bool = False) -> Vehicle | None:
        """
        Find the vehicle in the database that best matches the user's description.
        
        Process:
        1. Normalize the user's description to standard format
        2. Query vector database for similar vehicles
        3. Filter results by confidence thresholds
        4. Apply rules based on strict mode:
           - Non-strict: Return immediately if single high-confidence match
           - Strict: Always verify with LLM
        5. If no results or LLM can't determine, return None
        
        Args:
            description: The raw vehicle description from the partner
            strict: If True, always verify with LLM regardless of confidence
            
        Returns:
            The matched Vehicle or None if no confident match is found
        """
        logger.info(f"get_similar_vehicles called with strict={strict}")
        
        # Step 1: Normalize the description
        normalized_description = self.normalization_service.normalize(description, full_normalization=True)
        logger.info(f"Normalized description: '{description}' -> '{normalized_description}'")
        
        # Step 2: Query vector database
        result = self.vector_service.query_by_description(normalized_description, config.vector_top_k)
        logger.info(f"Vector search result:\n{json.dumps(result, indent=2)}")

        # Check if result array is empty
        if not result or not result.get('result'):
            logger.info("No similar vehicles found - empty results")
            return None

        # Step 3: Filter results by thresholds
        high_confidence_results = [
            r for r in result['result']
            if r.get('score', 0) >= self.similarity_threshold
        ]
        
        best_effort_results = [
            r for r in result['result']
            if self.similarity_threshold_best_effort <= r.get('score', 0) < self.similarity_threshold
        ]
        
        logger.info(
            f"Filtered results: {len(high_confidence_results)} >= similarity_threshold({self.similarity_threshold}), "
            f"{len(best_effort_results)} in [similarity_threshold_best_effort({self.similarity_threshold_best_effort}), similarity_threshold)"
        )
        
        # Check if any results above best-effort threshold
        all_candidates = high_confidence_results + best_effort_results
        if not all_candidates:
            logger.info("No results above best-effort threshold")
            return None
        
        # Non-strict mode: Single high-confidence result, return immediately
        if not strict and len(high_confidence_results) == 1:
            id_crabi = high_confidence_results[0].get('id')
            score = high_confidence_results[0].get('score')
            logger.info(f"Single high-confidence match found: {id_crabi} with score {score}")
            
            vehicle = self.vehicle_repository.get_by_crabi_id(id_crabi)
            if not vehicle:
                logger.warning(f"Vehicle with id_crabi {id_crabi} not found in database")
                return None
            return vehicle
        
        # Determine candidates for LLM verification
        if strict:
            # Strict mode: Use all candidates above best-effort threshold
            filtered_results = all_candidates
            logger.info(f"Strict mode: verifying all candidates ({len(filtered_results)}) with LLM")
        elif len(high_confidence_results) >= 2:
            filtered_results = high_confidence_results
            logger.info(f"Multiple high-confidence matches ({len(filtered_results)}), using LLM to disambiguate")
        else:
            filtered_results = all_candidates
            logger.info(f"Best-effort matches found ({len(filtered_results)}), using LLM to disambiguate")
        
        crabi_ids = [r.get('id') for r in filtered_results]
        logger.info(f"Candidates for LLM: {crabi_ids}")
        
        vehicles = self.vehicle_repository.get_by_crabi_ids(crabi_ids)
        
        if not vehicles:
            logger.warning("No vehicles found in database for the matched crabi_ids")
            return None
        
        # Prepare options for LLM
        options = [
            VehicleOption(id=v.id_crabi, description=v.description)
            for v in vehicles
        ]
        
        # Use LLM to determine the correct vehicle
        llm_response = self.llm_service.match_vehicle(
            user_description=description,  # Use original description for LLM
            options=options
        )
        
        logger.info(
            f"LLM response: selected_id={llm_response.selected_id}, "
            f"confidence={llm_response.confidence}, "
            f"reasoning={llm_response.reasoning}"
        )
        
        # Step 6: Return the vehicle selected by LLM or None
        if llm_response.selected_id is None:
            logger.info("LLM could not determine a confident match")
            return None
        
        # Find the vehicle with the selected id
        selected_vehicle = next(
            (v for v in vehicles if v.id_crabi == llm_response.selected_id),
            None
        )
        
        if not selected_vehicle:
            logger.warning(
                f"LLM selected id_crabi {llm_response.selected_id} but not found in vehicles list"
            )
            return None
        
        logger.info(f"Final match: {selected_vehicle.id_crabi} - {selected_vehicle.description}")
        return selected_vehicle
