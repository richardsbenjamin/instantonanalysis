from .resolvers import register_resolvers
from .schemas import register_schemas

def initialise_hydra_extensions():
    register_resolvers()
    register_schemas()

initialise_hydra_extensions()