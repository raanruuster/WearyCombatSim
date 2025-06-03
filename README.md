# TTRPG Combat Simulator

A Monte Carlo simulation tool for analyzing tabletop RPG combat encounters. This simulator helps game masters balance their encounters by running thousands of simulations and providing detailed statistics about likely outcomes.

## Features

### Party Configuration
- Support for 1-6 party members
- Three combat styles:
  - Light (1d6, 5 swings per round)
  - Medium (2d6, 2 swings per round)
  - Heavy (2d10, 1 swing per round)
- Customizable stats for each character:
  - HP and DR (Damage Reduction)
  - Proficiency and bonus damage
  - Custom damage dice and number of attacks
  - Stamina cost per attack

### Enemy Types
- Four enemy categories with customizable stats:
  - Minions (default: 5 HP, 1d4+1 damage)
  - Soldiers (default: 10 HP, 2d6 damage, 2 DR)
  - Elites (default: 20 HP, 2d6+3 damage)
  - Boss (default: 40 HP, 3d6+4 damage)

### Preset Encounters
- **Minion Horde**: 15 weak enemies testing AoE and crowd control
- **Elite Squad**: 4 tough soldiers with high DR
- **Glass Cannons**: 6 high-damage but fragile enemies
- **Boss Battle**: Classic boss fight with elite guards

### Combat Analysis
- Runs 10,000 simulations per analysis
- Detailed statistics including:
  - Victory and TPK rates
  - Partial party casualty analysis
  - Combat duration statistics
  - Average damage per round
- Detailed combat log showing:
  - Individual attack rolls and damage
  - DR calculations
  - HP tracking
  - Critical moments (defeats, victories)

## How to Use

1. Configure your party members in the sidebar
2. Either:
   - Select a preset encounter, or
   - Configure custom enemy groups
3. Click "Run Simulation" to analyze the encounter
4. Review the statistics and sample combat log
5. Adjust and rerun as needed for balancing

## Technical Details

Built with:
- Python 3.12
- Streamlit
- NumPy
- Pandas

The simulator uses Monte Carlo methods to run 10,000 iterations of each combat scenario, providing statistically significant results for encounter analysis.

## Live Demo

You can try the simulator at: [Your Streamlit URL will go here]

## Local Development

To run locally:

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run encounter_sim.py
   ```

## Contributing

Feel free to open issues or submit pull requests with improvements or bug fixes.

## License

MIT License - Feel free to use and modify for your own games! 