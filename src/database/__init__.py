
from .graph_models.maintenance_personnel import MaintenanceWorker, MaintenanceRecord, MaintenancePerformance, SkillAssessment, SkillAssessResult

from neomodel import config, db

def connect_to_neo4j(address: str, username: str, password: str):
	r"""
	address is like: 'localhost:7687'
	"""
	config.DATABASE_URL = f'bolt://{ username }:{ password }@{ address }'
	db.set_connection(f'bolt://{ username }:{ password }@{ address }')