import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import subprocess
import sys

# Player and enemy color schemes
PLAYER_COLORS = ["blue", "green", "orange", "red", "violet", "rainbow"]
ENEMY_TYPE_COLORS = {
    "Minion": "gray",
    "Soldier": "blue",
    "Elite": "violet",
    "Boss": "red",
}


def get_player_color(player_idx: int) -> str:
    """Get consistent color for a player."""
    return PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]


def get_enemy_color(enemy_type: str) -> str:
    """Get color for enemy type."""
    return ENEMY_TYPE_COLORS.get(enemy_type, "gray")


def format_player_name(player_idx: int, style: str) -> str:
    """Format player name with color."""
    color = get_player_color(player_idx)
    return f":{color}[**Player {player_idx + 1}**] ({style})"


def format_enemy_name(enemy_name: str) -> str:
    """Format enemy name with color based on type."""
    enemy_type = enemy_name.split("_")[0]
    color = get_enemy_color(enemy_type)
    return f":{color}[**{enemy_name}**]"


# comment
@dataclass
class Character:
    """Represents a party member with combat stats."""

    style: str
    proficiency: int
    bonus_damage: int
    base_hp: int
    dr: int
    damage_dice: str
    max_swings: int
    stamina_cost: int

    def roll_damage(self, rng: np.random.Generator) -> int:
        """Roll damage for one attack."""
        dice_parts = self.damage_dice.split("d")
        num_dice = int(dice_parts[0])
        die_size = int(dice_parts[1])
        return rng.integers(1, die_size + 1, num_dice).sum()


@dataclass
class Enemy:
    """Represents an enemy combatant."""

    hp: int
    dr: int
    damage_dice: str  # Format: "XdY+Z"

    def roll_damage(self, rng: np.random.Generator) -> int:
        """Roll damage for one attack."""
        # Split into dice part and bonus part
        if "+" in self.damage_dice:
            dice_part, bonus_part = self.damage_dice.split("+")
            bonus = int(bonus_part)
        else:
            dice_part = self.damage_dice
            bonus = 0

        # Parse dice
        dice_parts = dice_part.split("d")
        num_dice = int(dice_parts[0])
        die_size = int(dice_parts[1])

        return rng.integers(1, die_size + 1, num_dice).sum() + bonus


def format_hp_change(old_hp: int, new_hp: int) -> str:
    """Format HP change with color."""
    if new_hp > old_hp:
        return f"**{old_hp}** → :green[**{new_hp}**]"
    elif new_hp < old_hp:
        return f"**{old_hp}** → :red[**{new_hp}**]"
    else:
        return f"**{old_hp}** → **{new_hp}**"


def format_damage_calc(raw: int, prof: int, bonus: int, total: int) -> str:
    """Format damage calculation with color."""
    return f":blue[{raw}] + {prof} (prof) + {bonus} (bonus) = :blue[**{total}**]"


def add_custom_css():
    """Add custom CSS for better visual styling."""
    st.markdown(
        """
    <style>
    .round-container {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .combat-log {
        font-family: 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        max-height: 600px;
        overflow-y: auto;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def simulate_encounter(
    party: List[Character], enemies: List[Enemy], n_iters: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Run Monte Carlo simulation of encounters."""
    results = []
    combat_logs = []
    turn_economy_data = []

    # Create enemy names based on their position in the enemy groups list
    def get_enemy_name(enemy_idx: int, enemies: List[Enemy]) -> str:
        """Generate a unique enemy name based on their type and count."""
        count = 1
        for i in range(enemy_idx):
            if (
                enemies[i].hp == enemies[enemy_idx].hp
                and enemies[i].dr == enemies[enemy_idx].dr
                and enemies[i].damage_dice == enemies[enemy_idx].damage_dice
            ):
                count += 1

        # Determine enemy type based on their stats
        if enemies[enemy_idx].hp <= 5:
            enemy_type = "Minion"
        elif enemies[enemy_idx].hp <= 10:
            enemy_type = "Soldier"
        elif enemies[enemy_idx].hp <= 20:
            enemy_type = "Elite"
        else:
            enemy_type = "Boss"

        return f"{enemy_type}_{count}"

    def get_initiative_order(
        party_size: int, enemy_size: int, rng: np.random.Generator
    ) -> List[Tuple[str, int]]:
        """Generate initiative order for all combatants."""
        all_combatants = [("player", i) for i in range(party_size)] + [
            ("enemy", i) for i in range(enemy_size)
        ]
        initiatives = [(c, rng.integers(1, 21)) for c in all_combatants]
        sorted_initiatives = sorted(initiatives, key=lambda x: x[1], reverse=True)
        return [combatant for combatant, _ in sorted_initiatives]

    for i in range(n_iters):
        # Reset state
        party_hp = [c.base_hp for c in party]
        enemy_hp = [e.hp for e in enemies]
        enemy_names = [get_enemy_name(idx, enemies) for idx in range(len(enemies))]
        round_num = 0
        combat_log = []
        result_recorded = False
        first_casualty_round = None  # Track when first player falls

        # Turn economy tracking
        turn_data = {
            "player_actions": 0,
            "enemy_actions": 0,
            "total_player_damage": 0,
            "total_enemy_damage": 0,
            "stamina_spent": 0,
            "rounds": 0,
        }

        while round_num < 100:
            round_num += 1
            round_log = [f"## 🎲 Round {round_num}"]
            turn_data["rounds"] = round_num

            # Determine initiative order for this round
            initiative_order = get_initiative_order(len(party), len(enemies), rng)

            # Each combatant takes their turn in initiative order
            for combatant_type, idx in initiative_order:
                if combatant_type == "player":
                    if party_hp[idx] <= 0:
                        continue

                    char = party[idx]
                    max_possible_swings = min(
                        char.max_swings, party_hp[idx] // char.stamina_cost
                    )

                    if max_possible_swings > 1:
                        chosen_swings = rng.integers(1, max_possible_swings + 1)
                    else:
                        chosen_swings = max_possible_swings

                    for swing in range(chosen_swings):
                        if party_hp[idx] >= char.stamina_cost:
                            old_hp = party_hp[idx]
                            party_hp[idx] -= char.stamina_cost
                            turn_data["stamina_spent"] += char.stamina_cost

                            for e_idx in range(len(enemy_hp)):
                                if enemy_hp[e_idx] > 0:
                                    raw_damage = char.roll_damage(rng)
                                    total_damage = (
                                        raw_damage
                                        + char.proficiency
                                        + char.bonus_damage
                                    )
                                    net_damage = max(
                                        total_damage - enemies[e_idx].dr, 0
                                    )
                                    old_enemy_hp = enemy_hp[e_idx]
                                    enemy_hp[e_idx] -= net_damage

                                    # Track turn economy
                                    turn_data["player_actions"] += 1
                                    turn_data["total_player_damage"] += net_damage

                                    # Log the attack with colors
                                    if swing == 0 and max_possible_swings > 1:
                                        action = f"### 🎯 {format_player_name(idx, char.style)} chooses to make :blue[{chosen_swings}] attacks this round"
                                        round_log.append(action)

                                    action = [
                                        f"#### ⚔️ {format_player_name(idx, char.style)} attacks {format_enemy_name(enemy_names[e_idx])}:",
                                        f"- Roll: {format_damage_calc(raw_damage, char.proficiency, char.bonus_damage, total_damage)}",
                                        f"- DR {enemies[e_idx].dr} reduces to :orange[**{net_damage}**] damage",
                                    ]

                                    if enemy_hp[e_idx] <= 0:
                                        action.append(
                                            f"- 💀 {format_enemy_name(enemy_names[e_idx])} perishes! ({format_hp_change(old_enemy_hp, 0)})"
                                        )
                                    else:
                                        action.append(
                                            f"- {format_enemy_name(enemy_names[e_idx])} HP: {format_hp_change(old_enemy_hp, enemy_hp[e_idx])}"
                                        )

                                    action.append(
                                        f"- {format_player_name(idx, char.style)} HP: {format_hp_change(old_hp, party_hp[idx])} (after stamina cost)"
                                    )
                                    round_log.append("\n".join(action))
                                    break

                            if all(hp <= 0 for hp in enemy_hp) and not result_recorded:
                                round_log.append("## 🎉 All enemies defeated!")
                                combat_log.append("\n\n".join(round_log))
                                results.append(
                                    {
                                        "iteration": i,
                                        "outcome": "victory",
                                        "rounds": round_num,
                                        "surviving_party": sum(
                                            1 for hp in party_hp if hp > 0
                                        ),
                                        "first_casualty_round": first_casualty_round,
                                    }
                                )
                                result_recorded = True
                                break
                else:
                    # Enemy's turn
                    if enemy_hp[idx] <= 0:
                        continue

                    alive_party = [p_idx for p_idx, hp in enumerate(party_hp) if hp > 0]
                    if alive_party:
                        target_idx = rng.choice(alive_party)
                        raw_damage = enemies[idx].roll_damage(rng)
                        net_damage = max(raw_damage - party[target_idx].dr, 0)
                        old_hp = party_hp[target_idx]
                        party_hp[target_idx] -= net_damage

                        # Track turn economy
                        turn_data["enemy_actions"] += 1
                        turn_data["total_enemy_damage"] += net_damage

                        # Log the attack with colors
                        action = [
                            f"#### 🗡️ {format_enemy_name(enemy_names[idx])} attacks {format_player_name(target_idx, party[target_idx].style)}:",
                            f"- Roll: :blue[{raw_damage}]",
                            f"- DR {party[target_idx].dr} reduces to :orange[**{net_damage}**] damage",
                        ]

                        if party_hp[target_idx] <= 0:
                            action.append(
                                f"- 💀 {format_player_name(target_idx, party[target_idx].style)} is defeated! ({format_hp_change(old_hp, 0)})"
                            )
                            # Record first casualty if not already recorded
                            if first_casualty_round is None:
                                first_casualty_round = round_num
                        else:
                            action.append(
                                f"- {format_player_name(target_idx, party[target_idx].style)} HP: {format_hp_change(old_hp, party_hp[target_idx])}"
                            )

                        round_log.append("\n".join(action))

            if result_recorded:
                break

            combat_log.append("\n\n".join(round_log))

            # Check if party can still attack
            can_attack = any(
                party_hp[idx] >= char.stamina_cost
                for idx, char in enumerate(party)
                if party_hp[idx] > 0
            )

            if not can_attack and not result_recorded:
                if all(hp <= 0 for hp in party_hp):
                    round_log.append("## ☠️ Total party kill!")
                else:
                    round_log.append("## 😫 Party exhausted!")
                combat_log.append("\n\n".join(round_log))
                if all(hp <= 0 for hp in party_hp):
                    results.append(
                        {
                            "iteration": i,
                            "outcome": "tpk",
                            "rounds": round_num,
                            "surviving_enemies": sum(1 for hp in enemy_hp if hp > 0),
                            "first_casualty_round": first_casualty_round,
                        }
                    )
                else:
                    results.append(
                        {
                            "iteration": i,
                            "outcome": "exhaustion",
                            "rounds": round_num,
                            "surviving_party": sum(1 for hp in party_hp if hp > 0),
                            "surviving_enemies": sum(1 for hp in enemy_hp if hp > 0),
                            "first_casualty_round": first_casualty_round,
                        }
                    )
                result_recorded = True
                break

        if round_num >= 100 and not result_recorded:
            round_log.append("Combat timeout!")
            combat_log.append("\n\n".join(round_log))
            results.append(
                {
                    "iteration": i,
                    "outcome": "timeout",
                    "rounds": 100,
                }
            )
            result_recorded = True

        # Store turn economy data
        turn_data["first_casualty_round"] = first_casualty_round
        turn_economy_data.append(turn_data.copy())
        combat_logs.append("\n\n".join(combat_log))

    # Store additional data in results
    if results:
        results[0]["combat_logs"] = combat_logs
        results[0]["turn_economy_data"] = turn_economy_data

    return pd.DataFrame(results)


def calculate_average_dpr(
    party: List[Character], rng: np.random.Generator, n_samples: int = 1000
) -> float:
    """Calculate average DPR for the party."""
    total_damage = 0
    for _ in range(n_samples):
        for char in party:
            if char.base_hp >= char.stamina_cost:
                swings = min(char.max_swings, char.base_hp // char.stamina_cost)
                for _ in range(swings):
                    total_damage += (
                        char.roll_damage(rng) + char.proficiency + char.bonus_damage
                    )
    return total_damage / n_samples


def main():
    st.set_page_config(page_title="TTRPG Encounter Simulator", layout="wide")
    add_custom_css()  # Add custom CSS
    st.title("Turn-Based Combat Encounter Simulator")

    # Sidebar controls
    with st.sidebar:
        st.header("Encounter Parameters")

        party_size = st.number_input("Party Size", min_value=1, max_value=6, value=4)

        st.subheader("Party Configuration")
        party = []

        style_defaults = {
            "Light": {"dice": "1d6", "swings": 5, "stamina": 1},
            "Medium": {"dice": "2d6", "swings": 2, "stamina": 1},
            "Heavy": {"dice": "2d10", "swings": 1, "stamina": 1},
        }

        # Default party member configurations
        default_party = [
            {"style": "Light", "hp": 20, "dr": 2, "prof": 2},
            {"style": "Medium", "hp": 20, "dr": 4, "prof": 2},
            {"style": "Medium", "hp": 20, "dr": 4, "prof": 2},
            {"style": "Heavy", "hp": 20, "dr": 6, "prof": 2},
        ]

        for i in range(party_size):
            with st.expander(f"Party Member {i + 1}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    default_style = (
                        default_party[i]["style"]
                        if i < len(default_party)
                        else "Medium"
                    )
                    style = st.selectbox(
                        f"Style",
                        ["Light", "Medium", "Heavy"],
                        index=["Light", "Medium", "Heavy"].index(default_style),
                        key=f"style_{i}",
                    )
                    proficiency = st.number_input(
                        "Proficiency",
                        min_value=0,
                        value=default_party[i]["prof"] if i < len(default_party) else 2,
                        key=f"prof_{i}",
                    )
                    bonus_damage = st.number_input(
                        "Bonus Damage", min_value=0, value=0, key=f"bonus_{i}"
                    )
                with col2:
                    base_hp = st.number_input(
                        "Base HP",
                        min_value=1,
                        value=default_party[i]["hp"] if i < len(default_party) else 20,
                        key=f"hp_{i}",
                    )
                    dr = st.number_input(
                        "DR",
                        min_value=0,
                        value=default_party[i]["dr"] if i < len(default_party) else 0,
                        key=f"dr_{i}",
                    )

                st.text("Override defaults:")
                damage_dice = st.text_input(
                    "Damage Dice", value=style_defaults[style]["dice"], key=f"dice_{i}"
                )
                max_swings = st.number_input(
                    "Max Swings",
                    min_value=1,
                    value=style_defaults[style]["swings"],
                    key=f"swings_{i}",
                )
                stamina_cost = st.number_input(
                    "Stamina Cost",
                    min_value=1,
                    value=style_defaults[style]["stamina"],
                    key=f"stamina_{i}",
                )

                party.append(
                    Character(
                        style,
                        proficiency,
                        bonus_damage,
                        base_hp,
                        dr,
                        damage_dice,
                        max_swings,
                        stamina_cost,
                    )
                )

        st.sidebar.subheader("Preset Encounters")
        preset_col1, preset_col2 = st.sidebar.columns(2)

        # Define preset encounters
        presets = {
            "Minion Horde": {
                "description": "15 weak enemies with low HP",
                "setup": [
                    {"type": "Minions", "count": 15, "hp": 4, "dr": 0, "damage": "1d4"},
                    {
                        "type": "Soldiers",
                        "count": 0,
                        "hp": 10,
                        "dr": 2,
                        "damage": "2d6",
                    },
                    {
                        "type": "Elites",
                        "count": 0,
                        "hp": 20,
                        "dr": 3,
                        "damage": "2d6+3",
                    },
                    {"type": "Boss", "count": 0, "hp": 40, "dr": 4, "damage": "3d6+4"},
                ],
            },
            "Elite Squad": {
                "description": "4 tough soldiers with good DR",
                "setup": [
                    {
                        "type": "Minions",
                        "count": 0,
                        "hp": 5,
                        "dr": 0,
                        "damage": "1d4+1",
                    },
                    {
                        "type": "Soldiers",
                        "count": 4,
                        "hp": 15,
                        "dr": 4,
                        "damage": "2d6+2",
                    },
                    {
                        "type": "Elites",
                        "count": 0,
                        "hp": 20,
                        "dr": 3,
                        "damage": "2d6+3",
                    },
                    {"type": "Boss", "count": 0, "hp": 40, "dr": 4, "damage": "3d6+4"},
                ],
            },
            "Glass Cannons": {
                "description": "6 high-damage, low-HP enemies",
                "setup": [
                    {
                        "type": "Minions",
                        "count": 0,
                        "hp": 5,
                        "dr": 0,
                        "damage": "1d4+1",
                    },
                    {"type": "Soldiers", "count": 6, "hp": 8, "dr": 0, "damage": "3d6"},
                    {
                        "type": "Elites",
                        "count": 0,
                        "hp": 20,
                        "dr": 3,
                        "damage": "2d6+3",
                    },
                    {"type": "Boss", "count": 0, "hp": 40, "dr": 4, "damage": "3d6+4"},
                ],
            },
            "Boss Battle": {
                "description": "1 tough boss with 2 elite guards",
                "setup": [
                    {
                        "type": "Minions",
                        "count": 0,
                        "hp": 5,
                        "dr": 0,
                        "damage": "1d4+1",
                    },
                    {
                        "type": "Soldiers",
                        "count": 0,
                        "hp": 10,
                        "dr": 2,
                        "damage": "2d6",
                    },
                    {
                        "type": "Elites",
                        "count": 2,
                        "hp": 20,
                        "dr": 3,
                        "damage": "2d6+3",
                    },
                    {"type": "Boss", "count": 1, "hp": 50, "dr": 5, "damage": "4d6+4"},
                ],
            },
        }

        # Create buttons for each preset
        if preset_col1.button(
            "🗡️ Minion Horde", help=presets["Minion Horde"]["description"]
        ):
            st.session_state.selected_preset = "Minion Horde"
        if preset_col2.button(
            "🛡️ Elite Squad", help=presets["Elite Squad"]["description"]
        ):
            st.session_state.selected_preset = "Elite Squad"
        if preset_col1.button(
            "⚔️ Glass Cannons", help=presets["Glass Cannons"]["description"]
        ):
            st.session_state.selected_preset = "Glass Cannons"
        if preset_col2.button(
            "👑 Boss Battle", help=presets["Boss Battle"]["description"]
        ):
            st.session_state.selected_preset = "Boss Battle"

        st.sidebar.markdown("---")
        st.sidebar.subheader("Enemy Groups")

        # Initialize session state for presets if not exists
        if "selected_preset" not in st.session_state:
            st.session_state.selected_preset = None

        enemies = []  # Initialize enemies list

        enemy_groups = [
            {"name": "Minions", "default_dice": "1d4+1", "default_hp": 5},
            {"name": "Soldiers", "default_dice": "2d6", "default_hp": 10},
            {"name": "Elites", "default_dice": "2d6+3", "default_hp": 20},
            {"name": "Boss", "default_dice": "3d6+4", "default_hp": 40},
        ]

        for i, group in enumerate(enemy_groups):
            with st.expander(f"{group['name']} Configuration"):
                col1, col2 = st.columns(2)

                # Get preset values if a preset is selected
                preset_values = None
                if st.session_state.selected_preset:
                    preset_values = presets[st.session_state.selected_preset]["setup"][
                        i
                    ]

                with col1:
                    count = st.number_input(
                        f"Number of {group['name']}",
                        min_value=0,
                        max_value=20,
                        value=preset_values["count"]
                        if preset_values
                        else (5 if i == 0 else (3 if i == 1 else 0)),
                        key=f"enemy_count_{i}",
                    )
                    hp = st.number_input(
                        "HP per Enemy",
                        min_value=1,
                        value=preset_values["hp"]
                        if preset_values
                        else group["default_hp"],
                        key=f"enemy_hp_{i}",
                    )
                with col2:
                    dr = st.number_input(
                        "DR",
                        min_value=0,
                        value=preset_values["dr"]
                        if preset_values
                        else (2 if i == 1 else 0),
                        key=f"enemy_dr_{i}",
                    )
                    damage_dice = st.text_input(
                        "Damage Dice",
                        value=preset_values["damage"]
                        if preset_values
                        else group["default_dice"],
                        key=f"enemy_dice_{i}",
                    )

                # Add enemies from this group
                for _ in range(count):
                    enemies.append(Enemy(hp, dr, damage_dice))

        st.subheader("Simulation Settings")
        seed = st.number_input("RNG Seed", min_value=0, value=42)

        run_sim = st.button("Run Simulation", type="primary")

    # Main panel
    if run_sim:
        if not enemies:
            st.error("Please add at least one enemy to simulate!")
            return

        rng = np.random.default_rng(seed)

        with st.spinner("Running simulation..."):
            results_df = simulate_encounter(party, enemies, 10000, rng)

        # Calculate metrics
        victory_df = results_df[results_df["outcome"] == "victory"]
        tpk_df = results_df[results_df["outcome"] == "tpk"]
        exhaustion_df = results_df[results_df["outcome"] == "exhaustion"]

        # Display average DPR
        st.subheader("Average Damage Per Round")
        party_dpr = calculate_average_dpr(party, rng)

        # Calculate enemy DPR using their damage dice
        enemy_dpr = 0
        for enemy in enemies:
            total = 0
            n_samples = 1000
            for _ in range(n_samples):
                total += enemy.roll_damage(rng)
            enemy_dpr += total / n_samples

        col1, col2 = st.columns(2)
        col1.metric("Party Average DPR", f"{party_dpr:.1f}")
        col2.metric("Enemy Total DPR", f"{enemy_dpr:.1f}")

        # Outcome summary with percentages
        st.subheader("Combat Outcomes")
        total_sims = len(results_df)

        col1, col2 = st.columns(2)
        victory_pct = len(victory_df) / total_sims * 100
        tpk_pct = len(tpk_df) / total_sims * 100

        col1.metric("Victories", f"{len(victory_df):,} ({victory_pct:.1f}%)")
        col2.metric("TPKs", f"{len(tpk_df):,} ({tpk_pct:.1f}%)")

        # Analyze partial party casualties
        st.subheader("Party Casualties Analysis")

        # For victories, calculate how many party members were lost
        if len(victory_df) > 0:
            party_size = len(party)
            casualties_in_victories = {}
            for i in range(
                1, party_size
            ):  # 1 to party_size-1 casualties (not 0 or TPK)
                victories_with_n_casualties = victory_df[
                    victory_df["surviving_party"] == (party_size - i)
                ]
                if len(victories_with_n_casualties) > 0:
                    pct = len(victories_with_n_casualties) / total_sims * 100
                    casualties_in_victories[i] = pct

            if casualties_in_victories:
                st.write("In successful battles:")
                for casualties, percentage in casualties_in_victories.items():
                    st.write(
                        f"- {percentage:.1f}% saw {casualties} {'player' if casualties == 1 else 'players'} fall"
                    )
            else:
                st.write("In successful battles, the party never suffered casualties.")

        # Combat Duration Statistics
        st.subheader("Combat Duration Statistics")

        if len(victory_df) > 0:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Victory Round Distribution")
                hist_data = victory_df["rounds"].value_counts().sort_index()
                st.bar_chart(hist_data)
                median_victory = victory_df["rounds"].median()
                mean_victory = victory_df["rounds"].mean()
                st.metric("Median Victory Round", f"{median_victory:.1f}")
                st.metric("Mean Victory Round", f"{mean_victory:.1f}")

            with col2:
                if len(tpk_df) > 0:
                    st.write("TPK Round Distribution")
                    hist_data = tpk_df["rounds"].value_counts().sort_index()
                    st.bar_chart(hist_data)
                    st.metric("Median TPK Round", f"{tpk_df['rounds'].median():.1f}")
                    st.metric("Mean TPK Round", f"{tpk_df['rounds'].mean():.1f}")

            with col3:
                st.write("First Casualty Analysis")
                casualty_data = results_df[results_df["first_casualty_round"].notna()]
                if not casualty_data.empty:
                    hist_data = (
                        casualty_data["first_casualty_round"]
                        .value_counts()
                        .sort_index()
                    )
                    st.bar_chart(hist_data)
                    st.metric(
                        "Avg Round of First Casualty",
                        f"{casualty_data['first_casualty_round'].mean():.1f}",
                    )
                    st.metric(
                        "Median Round of First Casualty",
                        f"{casualty_data['first_casualty_round'].median():.1f}",
                    )
                else:
                    st.metric("First Casualty", "No casualties")

        # Encounter Balance Verdict
        st.subheader("Encounter Balance Analysis")

        # Calculate detailed statistics
        if len(victory_df) > 0:
            victory_rounds = victory_df["rounds"]
            round_stats = {
                "min": victory_rounds.min(),
                "q1": victory_rounds.quantile(0.25),
                "median": victory_rounds.median(),
                "q3": victory_rounds.quantile(0.75),
                "max": victory_rounds.max(),
            }

            # Determine verdict based on victory probability
            if victory_pct >= 80:
                verdict = "Easy Encounter"
                color = "🟢"
                details = "High probability of party victory"
            elif victory_pct >= 60:
                verdict = "Moderate Challenge"
                color = "🟡"
                details = "Party favored but not guaranteed"
            elif victory_pct >= 40:
                verdict = "Hard Challenge"
                color = "🟠"
                details = "Balanced but difficult"
            elif victory_pct >= 20:
                verdict = "Very Hard"
                color = "🔴"
                details = "Party disadvantaged but victory possible"
            else:
                verdict = "Extreme Challenge"
                color = "⚫"
                details = "Victory unlikely"

            st.markdown(f"### {color} {verdict}")
            st.write(details)

            # Display round statistics
            st.write("Combat Duration Analysis:")
            st.write(f"- Quickest Victory: Round {round_stats['min']}")
            st.write(
                f"- Typical Range: Round {round_stats['q1']:.1f} to {round_stats['q3']:.1f}"
            )
            st.write(f"- Longest Victory: Round {round_stats['max']}")

            if len(tpk_df) > 0:
                st.write(
                    f"- Party Wipe Risk: {tpk_pct:.1f}% chance, typically around round {tpk_df['rounds'].median():.1f}"
                )
        else:
            st.markdown("### ⚫ Unwinnable")
            st.write("Party unable to achieve victory in any simulation")
            if len(tpk_df) > 0:
                st.write(f"Average TPK occurs on round {tpk_df['rounds'].mean():.1f}")

        # NEW: Survival Probability Graph
        st.subheader("📊 Survival Probability by Round")

        if len(victory_df) > 0:
            # Calculate survival probability by round
            max_rounds = victory_df["rounds"].max()
            survival_prob = []
            rounds_list = list(range(1, max_rounds + 1))

            for round_num in rounds_list:
                victories_by_round = len(victory_df[victory_df["rounds"] <= round_num])
                prob = victories_by_round / len(results_df) * 100
                survival_prob.append(prob)

            # Create DataFrame for plotting
            survival_df = pd.DataFrame(
                {"Round": rounds_list, "Victory Probability %": survival_prob}
            )

            st.line_chart(survival_df.set_index("Round"))

            # Add some key statistics
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "50% Victory Chance by Round",
                f"{rounds_list[next((i for i, p in enumerate(survival_prob) if p >= 50), len(rounds_list) - 1)] if survival_prob else 'N/A'}",
            )
            col2.metric(
                "90% Victory Chance by Round",
                f"{rounds_list[next((i for i, p in enumerate(survival_prob) if p >= 90), len(rounds_list) - 1)] if any(p >= 90 for p in survival_prob) else 'Never'}",
            )
            col3.metric("Peak Victory Rate", f"{max(survival_prob):.1f}%")

        # NEW: Turn Economy Analysis
        st.subheader("⚖️ Turn Economy Analysis")

        if "turn_economy_data" in results_df.iloc[0]:
            turn_data = results_df.iloc[0]["turn_economy_data"]

            # Calculate averages
            avg_player_actions = np.mean([d["player_actions"] for d in turn_data])
            avg_enemy_actions = np.mean([d["enemy_actions"] for d in turn_data])
            avg_player_damage = np.mean([d["total_player_damage"] for d in turn_data])
            avg_enemy_damage = np.mean([d["total_enemy_damage"] for d in turn_data])
            avg_stamina_spent = np.mean([d["stamina_spent"] for d in turn_data])
            avg_rounds = np.mean([d["rounds"] for d in turn_data])

            # Calculate efficiency metrics
            damage_per_action = (
                avg_player_damage / avg_player_actions if avg_player_actions > 0 else 0
            )
            damage_per_stamina = (
                avg_player_damage / avg_stamina_spent if avg_stamina_spent > 0 else 0
            )
            actions_per_round = avg_player_actions / avg_rounds if avg_rounds > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg Player Actions", f"{avg_player_actions:.1f}")
            col2.metric("Avg Enemy Actions", f"{avg_enemy_actions:.1f}")
            col3.metric(
                "Action Economy Ratio",
                f"{avg_player_actions / avg_enemy_actions:.2f}"
                if avg_enemy_actions > 0
                else "∞",
            )
            col4.metric("Actions per Round", f"{actions_per_round:.1f}")

            st.subheader("📈 Efficiency Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Damage per Action", f"{damage_per_action:.1f}")
            col2.metric("Damage per Stamina", f"{damage_per_stamina:.1f}")
            col3.metric(
                "Stamina Efficiency",
                f"{(avg_player_damage / avg_stamina_spent) * 100:.1f}%"
                if avg_stamina_spent > 0
                else "N/A",
            )

        # Update combat log display with styling
        st.sidebar.markdown("---")
        with st.sidebar.expander("📜 Sample Combat Log", expanded=False):
            if "combat_logs" in results_df.iloc[0]:
                combat_logs = results_df.iloc[0]["combat_logs"]
                sample_idx = rng.integers(0, len(combat_logs))

                # Apply styling to combat log
                st.markdown('<div class="round-container">', unsafe_allow_html=True)
                st.markdown(combat_logs[sample_idx])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("No combat log available")

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Raw Simulation Data",
            data=csv,
            file_name="encounter_simulation_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
