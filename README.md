# DroneZ

DroneZ is an OpenEnv-compliant environment for training and evaluating an LLM agent as a **fleet operations controller** for autonomous urban delivery drones.

The environment focuses on mission-level decision making under disruption:
- heterogeneous drones
- order prioritization and reassignment
- charging tradeoffs
- weather drift and no-fly constraints
- delivery failures and recovery actions
- interpretable reward components

## Source of truth
The primary project specifications live in:
- `context/urbanair_architecture.md`
- `context/urbanair_project_context.md`

## Current status
The repository is in bootstrap mode. The first milestone is establishing the repo structure, Claude guidance, config layout, and workflow needed to build the MVP vertical slice described in the source docs.
