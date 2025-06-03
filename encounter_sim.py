import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import subprocess
import sys
import math
import re  # For parsing damage dice
import copy  # For deep copying enemy lists

# Player and enemy color schemes for combat log
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
                                        action_msg = f"### 🎯 {format_player_name(idx, char.style)} chooses to make :blue[{chosen_swings}] attacks this round"
                                        round_log.append(action_msg)

                                    action_list = [
                                        f"#### ⚔️ {format_player_name(idx, char.style)} attacks {format_enemy_name(enemy_names[e_idx])}:",
                                        f"- Roll: {format_damage_calc(raw_damage, char.proficiency, char.bonus_damage, total_damage)}",
                                        f"- DR {enemies[e_idx].dr} reduces to :orange[**{net_damage}**] damage",
                                    ]

                                    if enemy_hp[e_idx] <= 0:
                                        action_list.append(
                                            f"- 💀 {format_enemy_name(enemy_names[e_idx])} perishes! ({format_hp_change(old_enemy_hp, 0)})"
                                        )
                                    else:
                                        action_list.append(
                                            f"- {format_enemy_name(enemy_names[e_idx])} HP: {format_hp_change(old_enemy_hp, enemy_hp[e_idx])}"
                                        )

                                    action_list.append(
                                        f"- {format_player_name(idx, char.style)} HP: {format_hp_change(old_hp, party_hp[idx])} (after stamina cost)"
                                    )
                                    round_log.append("\n".join(action_list))
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
                        action_list = [
                            f"#### 🗡️ {format_enemy_name(enemy_names[idx])} attacks {format_player_name(target_idx, party[target_idx].style)}:",
                            f"- Roll: :blue[{raw_damage}]",
                            f"- DR {party[target_idx].dr} reduces to :orange[**{net_damage}**] damage",
                        ]

                        if party_hp[target_idx] <= 0:
                            action_list.append(
                                f"- 💀 {format_player_name(target_idx, party[target_idx].style)} is defeated! ({format_hp_change(old_hp, 0)})"
                            )
                            # Record first casualty if not already recorded
                            if first_casualty_round is None:
                                first_casualty_round = round_num
                        else:
                            action_list.append(
                                f"- {format_player_name(target_idx, party[target_idx].style)} HP: {format_hp_change(old_hp, party_hp[target_idx])}"
                            )

                        round_log.append("\n".join(action_list))

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


# --- Helper function for adjusting damage dice string ---
def adjust_damage_dice(base_dice_str: str, bonus_change: int) -> str:
    match = re.fullmatch(r"(\d+d\d+)([+-]\d+)?", base_dice_str)
    if not match:
        return base_dice_str  # Should not happen with valid inputs

    dice_part = match.group(1)
    current_bonus_str = match.group(2)

    current_bonus = 0
    if current_bonus_str:
        current_bonus = int(current_bonus_str)

    new_bonus = current_bonus + bonus_change

    if new_bonus == 0:
        return dice_part
    elif new_bonus > 0:
        return f"{dice_part}+{new_bonus}"
    else:  # new_bonus < 0
        return f"{dice_part}{new_bonus}"  # e.g., 1d6-1


# --- Helper function to apply a single adjustment to an enemy list ---
def apply_single_adjustment(
    base_enemies: List[Enemy],
    template_enemy_defs: List[Dict],
    adjustment_rule: Dict,
    party_size: int,
    adjustment_direction: int,
) -> Tuple[List[Enemy], str]:
    """
    Applies a single adjustment to a list of enemies.
    adjustment_direction: 1 for positive change (increase metric like Vic%), -1 for negative (decrease metric)
    Returns the new list of enemies and a description of the change.
    """
    adjusted_enemies = copy.deepcopy(base_enemies)
    target_type = adjustment_rule["target_enemy_type"]
    param = adjustment_rule["param"]
    step = adjustment_rule["step"] * adjustment_direction  # Apply direction to step
    # max_increase = adjustment_rule["max_increase"] # Not fully used for clamping in this single step yet

    change_description = (
        f"Adjusting {target_type} {param} by {step} (Direction: {adjustment_direction})"
    )

    # Find the base definition for the target enemy type to get its original stats for cloning if needed
    base_def_for_target = next(
        (td for td in template_enemy_defs if td["type"] == target_type), None
    )
    if not base_def_for_target:
        return base_enemies, "Error: Target enemy type not in template for adjustment."

    if param == "count":
        current_count = sum(
            1 for e in adjusted_enemies if e.type_for_template == target_type
        )  # Assuming Enemy objects get a .type_for_template attr
        if step > 0:  # Increase count
            for _ in range(step):
                new_enemy = Enemy(
                    hp=base_def_for_target["hp"],
                    dr=base_def_for_target["dr"],
                    damage_dice=base_def_for_target["damage_dice"],
                )
                new_enemy.type_for_template = target_type  # Store its template type
                adjusted_enemies.append(new_enemy)
        elif step < 0:  # Decrease count
            removed_count = 0
            temp_list = []
            for e in reversed(adjusted_enemies):
                if e.type_for_template == target_type and removed_count < abs(step):
                    removed_count += 1
                else:
                    temp_list.append(e)
            adjusted_enemies = list(reversed(temp_list))
            if current_count + step < 0:  # Prevent negative counts overall
                change_description += " (limited by min count 0)"
                # This part needs more robust handling if we want to ensure min N enemies of a type

    elif param in ["hp", "dr"]:
        for enemy in adjusted_enemies:
            if (
                hasattr(enemy, "type_for_template")
                and enemy.type_for_template == target_type
            ):
                if param == "hp":
                    enemy.hp = max(1, enemy.hp + step)  # HP min 1
                elif param == "dr":
                    enemy.dr = max(0, enemy.dr + step)  # DR min 0

    elif param == "damage_bonus":
        for enemy in adjusted_enemies:
            if (
                hasattr(enemy, "type_for_template")
                and enemy.type_for_template == target_type
            ):
                enemy.damage_dice = adjust_damage_dice(enemy.damage_dice, step)

    return adjusted_enemies, change_description


def get_metric_from_df(results_df: pd.DataFrame, metric_name: str) -> float:
    if results_df is None or results_df.empty:
        return 0.0

    if metric_name == "Target Victory %":
        vict_df = results_df[results_df["outcome"] == "victory"]
        return (len(vict_df) / len(results_df) * 100) if len(results_df) > 0 else 0.0
    elif metric_name == "Target First Casualty %":
        if "first_casualty_round" in results_df.columns:
            fc_count = results_df["first_casualty_round"].notna().sum()
            return (fc_count / len(results_df) * 100) if len(results_df) > 0 else 0.0
        else:
            return 0.0  # Should not happen if tracking is on
    return 0.0


def summarize_enemies(enemy_list: List[Enemy], detail=False) -> List[str]:
    if not enemy_list:
        return ["No enemies"]

    summary_lines = []
    if detail:
        # For detailed summary, list each enemy with its template type
        for e in enemy_list:
            e_type = getattr(e, "type_for_template", "UnknownType")
            summary_lines.append(
                f"{e_type} (HP: {e.hp}, DR: {e.dr}, Dmg: {e.damage_dice})"
            )
    else:
        # For compact summary, group by type and exact stats
        counts = {}
        for e in enemy_list:
            e_type = getattr(e, "type_for_template", "UnknownType")
            # Create a key that uniquely identifies an enemy variant
            key = (e_type, e.hp, e.dr, e.damage_dice)
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            return ["No enemies to summarize"]

        for (e_type, hp, dr, dmg), count in counts.items():
            summary_lines.append(f"{count}x {e_type} (HP:{hp}, DR:{dr}, Dmg:'{dmg}')")

    return summary_lines


def main():
    st.set_page_config(page_title="TTRPG Encounter Simulator", layout="wide")
    add_custom_css()
    st.title("Turn-Based Combat Encounter Simulator")

    # --- Apply pending encounter configuration BEFORE creating any widgets ---
    if "builder_pending_config" in st.session_state:
        config = st.session_state.builder_pending_config
        for i in range(4):
            st.session_state[f"sim_enemy_count_{i}"] = config[f"count_{i}"]
            st.session_state[f"sim_enemy_hp_{i}"] = config[f"hp_{i}"]
            st.session_state[f"sim_enemy_dr_{i}"] = config[f"dr_{i}"]
            st.session_state[f"sim_enemy_dice_{i}"] = config[f"dice_{i}"]

        # Clear the pending config and show success message
        del st.session_state.builder_pending_config
        st.success(
            "✅ Encounter applied to simulator! Check the sidebar configuration."
        )

    # --- Initialize enemy widget defaults if not already set ---
    enemy_defaults = [
        {"count": 5, "hp": 5, "dr": 0, "dice": "1d4+1"},  # Minions
        {"count": 3, "hp": 10, "dr": 2, "dice": "2d6"},  # Soldiers
        {"count": 0, "hp": 20, "dr": 3, "dice": "2d6+3"},  # Elites
        {"count": 0, "hp": 40, "dr": 4, "dice": "3d6+4"},  # Boss
    ]
    for i in range(4):
        if f"sim_enemy_count_{i}" not in st.session_state:
            st.session_state[f"sim_enemy_count_{i}"] = enemy_defaults[i]["count"]
        if f"sim_enemy_hp_{i}" not in st.session_state:
            st.session_state[f"sim_enemy_hp_{i}"] = enemy_defaults[i]["hp"]
        if f"sim_enemy_dr_{i}" not in st.session_state:
            st.session_state[f"sim_enemy_dr_{i}"] = enemy_defaults[i]["dr"]
        if f"sim_enemy_dice_{i}" not in st.session_state:
            st.session_state[f"sim_enemy_dice_{i}"] = enemy_defaults[i]["dice"]

    # Sidebar controls - defined once, globally for the app
    with st.sidebar:
        st.header("Encounter Parameters")
        # Ensure party_size widget has its value stored in session_state for reliable access
        if "sim_party_size" not in st.session_state:
            st.session_state.sim_party_size = 4  # Default value if not set
        st.number_input("Party Size", min_value=1, max_value=6, key="sim_party_size")
        party_size = st.session_state.sim_party_size  # Use the value from session state

        st.subheader("Party Configuration")
        party = []
        style_defaults = {
            "Light": {"dice": "1d6", "swings": 5, "stamina": 1},
            "Medium": {"dice": "2d6", "swings": 2, "stamina": 1},
            "Heavy": {"dice": "2d10", "swings": 1, "stamina": 1},
        }
        default_party_configs = [
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
                        default_party_configs[i]["style"]
                        if i < len(default_party_configs)
                        else "Medium"
                    )
                    style = st.selectbox(
                        f"Style##{i}",
                        ["Light", "Medium", "Heavy"],
                        index=["Light", "Medium", "Heavy"].index(default_style),
                        key=f"sim_style_{i}",
                    )
                    proficiency = st.number_input(
                        "Proficiency",
                        min_value=0,
                        value=default_party_configs[i]["prof"]
                        if i < len(default_party_configs)
                        else 2,
                        key=f"sim_prof_{i}",
                    )
                    bonus_damage = st.number_input(
                        "Bonus Damage", min_value=0, value=0, key=f"sim_bonus_{i}"
                    )
                with col2:
                    base_hp = st.number_input(
                        "Base HP",
                        min_value=1,
                        value=default_party_configs[i]["hp"]
                        if i < len(default_party_configs)
                        else 20,
                        key=f"sim_hp_{i}",
                    )
                    dr = st.number_input(
                        "DR",
                        min_value=0,
                        value=default_party_configs[i]["dr"]
                        if i < len(default_party_configs)
                        else 0,
                        key=f"sim_dr_{i}",
                    )

                st.text("Override defaults:")
                damage_dice_input = st.text_input(
                    "Damage Dice",
                    value=style_defaults[style]["dice"],
                    key=f"sim_dice_{i}",
                )
                max_swings = st.number_input(
                    "Max Swings",
                    min_value=1,
                    value=style_defaults[style]["swings"],
                    key=f"sim_swings_{i}",
                )
                stamina_cost = st.number_input(
                    "Stamina Cost",
                    min_value=1,
                    value=style_defaults[style]["stamina"],
                    key=f"sim_stamina_{i}",
                )

                party.append(
                    Character(
                        style,
                        proficiency,
                        bonus_damage,
                        base_hp,
                        dr,
                        damage_dice_input,
                        max_swings,
                        stamina_cost,
                    )
                )

        st.sidebar.subheader("Preset Encounters")
        preset_col1, preset_col2 = st.sidebar.columns(2)
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
        if "selected_preset" not in st.session_state:
            st.session_state.selected_preset = None
        if preset_col1.button(
            "🗡️ Minion Horde",
            help=presets["Minion Horde"]["description"],
            key="preset_minion",
        ):
            st.session_state.selected_preset = "Minion Horde"
        if preset_col2.button(
            "🛡️ Elite Squad",
            help=presets["Elite Squad"]["description"],
            key="preset_elite",
        ):
            st.session_state.selected_preset = "Elite Squad"
        if preset_col1.button(
            "⚔️ Glass Cannons",
            help=presets["Glass Cannons"]["description"],
            key="preset_glass",
        ):
            st.session_state.selected_preset = "Glass Cannons"
        if preset_col2.button(
            "👑 Boss Battle",
            help=presets["Boss Battle"]["description"],
            key="preset_boss",
        ):
            st.session_state.selected_preset = "Boss Battle"

        st.sidebar.markdown("---")
        st.sidebar.subheader(
            "Enemy Groups (for Simulator Tab)"
        )  # Clarify this is for simulator
        enemies_for_simulator = []
        enemy_groups_config = [
            {"name": "Minions", "default_dice": "1d4+1", "default_hp": 5},
            {"name": "Soldiers", "default_dice": "2d6", "default_hp": 10},
            {"name": "Elites", "default_dice": "2d6+3", "default_hp": 20},
            {"name": "Boss", "default_dice": "3d6+4", "default_hp": 40},
        ]
        for i, group in enumerate(enemy_groups_config):
            with st.expander(f"{group['name']} Configuration"):
                col1, col2 = st.columns(2)
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
                        key=f"sim_enemy_count_{i}",
                    )
                    hp = st.number_input(
                        "HP per Enemy",
                        min_value=1,
                        key=f"sim_enemy_hp_{i}",
                    )
                with col2:
                    dr = st.number_input(
                        "DR",
                        min_value=0,
                        key=f"sim_enemy_dr_{i}",
                    )
                    damage_dice = st.text_input(
                        "Damage Dice",
                        key=f"sim_enemy_dice_{i}",
                    )
                for _ in range(count):
                    enemies_for_simulator.append(Enemy(hp, dr, damage_dice))

        st.subheader("Simulation Settings")
        if "sim_seed" not in st.session_state:
            st.session_state.sim_seed = 42
        st.number_input("RNG Seed", min_value=0, key="sim_seed")
        seed = st.session_state.sim_seed
        run_sim_button = st.button(
            "Run Simulation", type="primary", key="sim_run_button"
        )
        st.sidebar.markdown("---")
        st.sidebar.header("📜 Sample Combat Log")

    # --- Encounter Builder Template Definitions with Adjustment Priorities ---
    BUILDER_TEMPLATES = {
        "Lots of Minions": {
            "enemies": [
                {
                    "type": "Minion",
                    "count_func": lambda p_size: math.ceil(p_size * 2.5),
                    "hp": 10,
                    "dr": 0,
                    "damage_dice": "1d6",
                }
            ],
            "adjustment_priority": [
                {
                    "target_enemy_type": "Minion",
                    "param": "count",
                    "max_increase": 5,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Minion",
                    "param": "hp",
                    "max_increase": 3,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Minion",
                    "param": "dr",
                    "max_increase": 1,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Minion",
                    "param": "damage_bonus",
                    "max_increase": 2,
                    "step": 1,
                },
            ],
        },
        "Balanced Encounter": {
            "enemies": [
                {
                    "type": "Minion",
                    "count_func": lambda p_size: max(1, p_size - 1),
                    "hp": 10,
                    "dr": 0,
                    "damage_dice": "1d6",
                },
                {
                    "type": "Soldier",
                    "count_func": lambda p_size: math.ceil(p_size / 2.0),
                    "hp": 15,
                    "dr": 2,
                    "damage_dice": "2d6",
                },
                {
                    "type": "Elite",
                    "count_func": lambda p_size: 1,
                    "hp": 20,
                    "dr": 4,
                    "damage_dice": "2d10",
                },
            ],
            "adjustment_priority": [
                {
                    "target_enemy_type": "Minion",
                    "param": "count",
                    "max_increase": 5,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Soldier",
                    "param": "hp",
                    "max_increase": 5,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Elite",
                    "param": "dr",
                    "max_increase": 1,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Soldier",
                    "param": "damage_bonus",
                    "max_increase": 3,
                    "step": 1,
                },
            ],
        },
        "Boss with Minions": {
            "enemies": [
                {
                    "type": "Boss",
                    "count_func": lambda p_size: 1,
                    "hp": 30,
                    "dr": 6,
                    "damage_dice": "3d8+5",
                },
                {
                    "type": "Minion",
                    "count_func": lambda p_size: p_size + 1,
                    "hp": 10,
                    "dr": 0,
                    "damage_dice": "1d6",
                },
            ],
            "adjustment_priority": [
                {
                    "target_enemy_type": "Boss",
                    "param": "dr",
                    "max_increase": 2,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Minion",
                    "param": "count",
                    "max_increase": lambda p_size: math.ceil(p_size / 2.0),
                    "step": 1,
                },
                {
                    "target_enemy_type": "Boss",
                    "param": "hp",
                    "max_increase": 15,
                    "step": 1,
                },  # e.g., up to 50% base HP
                {
                    "target_enemy_type": "Boss",
                    "param": "damage_bonus",
                    "max_increase": 5,
                    "step": 1,
                },  # Adjusting the existing +5 bonus further, or adding if none
            ],
        },
        "One Tanky Enemy": {
            "enemies": [
                {
                    "type": "Elite",
                    "count_func": lambda p_size: 1,
                    "hp": 35,
                    "dr": 8,
                    "damage_dice": "1d6+2",
                }
            ],
            "adjustment_priority": [
                {
                    "target_enemy_type": "Elite",
                    "param": "hp",
                    "max_increase": 15,
                    "step": 1,
                },  # e.g. up to ~40-50% base HP
                {
                    "target_enemy_type": "Elite",
                    "param": "dr",
                    "max_increase": 2,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Elite",
                    "param": "damage_bonus",
                    "max_increase": 3,
                    "step": 1,
                },  # Adjusting the existing +2 bonus
            ],
        },
        "High DPS Enemies (Glass Cannons)": {
            "enemies": [
                {
                    "type": "Elite",
                    "count_func": lambda p_size: math.ceil(p_size / 2.0),
                    "hp": 8,
                    "dr": 0,
                    "damage_dice": "2d12",
                }
            ],
            "adjustment_priority": [
                {
                    "target_enemy_type": "Elite",
                    "param": "damage_bonus",
                    "max_increase": 5,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Elite",
                    "param": "count",
                    "max_increase": 1,
                    "step": 1,
                },
                {
                    "target_enemy_type": "Elite",
                    "param": "hp",
                    "max_increase": 5,
                    "step": 1,
                },  # Small HP increase for glass cannons
            ],
        },
    }
    # --- End of Encounter Builder Template Definitions ---

    TARGET_METRIC_TOLERANCE = 2.0  # Define the tolerance band (e.g., +/- 2%)

    tab1, tab2 = st.tabs(["⚔️ Combat Simulator", "🛠️ Encounter Builder"])

    with tab1:  ########## COMBAT SIMULATOR TAB ##########
        st.header("Simulate a Combat Encounter")

        # Initialize results_df and display variables to safe defaults
        results_df = pd.DataFrame()
        party_dpr_display = 0.0
        enemy_dpr_total_display = 0.0
        # Initialize other DataFrames that might be checked by display logic later
        victory_df = pd.DataFrame()
        tpk_df = pd.DataFrame()

        if run_sim_button:
            if not enemies_for_simulator:
                st.error("Please add at least one enemy to simulate!")
            else:
                rng = np.random.default_rng(seed)  # Define rng here
                with st.spinner("Running simulation..."):
                    # Simulate encounter
                    results_df_sim = simulate_encounter(
                        party, enemies_for_simulator, 10000, rng
                    )

                if not results_df_sim.empty:
                    results_df = results_df_sim  # Assign to the main results_df

                    # Calculate DPRs as rng and party/enemies are available here
                    party_dpr_display = calculate_average_dpr(party, rng)

                    current_enemy_dpr_total = 0
                    for enemy_obj in enemies_for_simulator:
                        temp_total = 0
                        n_samples_dpr_calc = 1000  # Use a different var name to avoid conflict if n_samples_dpr is used elsewhere
                        for _ in range(n_samples_dpr_calc):
                            temp_total += enemy_obj.roll_damage(rng)
                        current_enemy_dpr_total += temp_total / n_samples_dpr_calc
                    enemy_dpr_total_display = current_enemy_dpr_total

                    # Define victory_df and tpk_df here if results_df is populated
                    victory_df = results_df[results_df["outcome"] == "victory"]
                    tpk_df = results_df[results_df["outcome"] == "tpk"]
                else:
                    st.warning(
                        "Simulation ran but produced no results. Displaying defaults."
                    )
                    # results_df remains empty, dpr_display vars remain 0.0
                    # victory_df and tpk_df remain empty DataFrames
        else:  # if not run_sim_button
            st.info("Click 'Run Simulation' in the sidebar to view encounter results.")

        # --- Display sections ---
        # These sections will now use the initialized or populated variables.

        # Display average DPR
        st.subheader("Average Damage Per Round")
        col1_dpr, col2_dpr = st.columns(2)
        col1_dpr.metric("Party Average DPR", f"{party_dpr_display:.1f}")
        col2_dpr.metric("Enemy Total DPR", f"{enemy_dpr_total_display:.1f}")

        # Outcome summary with percentages
        st.subheader("Combat Outcomes")
        if not results_df.empty:
            total_sims = len(results_df)
            # victory_df and tpk_df are already defined if results_df is not empty
            col1_outcome, col2_outcome = st.columns(2)
            victory_pct = (len(victory_df) / total_sims * 100) if total_sims > 0 else 0
            tpk_pct = (len(tpk_df) / total_sims * 100) if total_sims > 0 else 0
            col1_outcome.metric(
                "Victories", f"{len(victory_df):,} ({victory_pct:.1f}%)"
            )
            col2_outcome.metric("TPKs", f"{len(tpk_df):,} ({tpk_pct:.1f}%)")
        else:
            st.write("Run simulation to see combat outcomes.")

        # Analyze partial party casualties
        st.subheader("Party Casualties Analysis")
        if (
            not victory_df.empty and "surviving_party" in victory_df.columns
        ):  # Check victory_df specifically
            party_size_val = len(party)
            casualties_in_victories = {}
            # Ensure total_sims is defined if results_df was not empty
            total_sims = len(results_df) if not results_df.empty else 0

            for i_cas in range(1, party_size_val):
                victories_with_n_casualties = victory_df[
                    victory_df["surviving_party"] == (party_size_val - i_cas)
                ]
                if len(victories_with_n_casualties) > 0 and total_sims > 0:
                    pct_cas = len(victories_with_n_casualties) / total_sims * 100
                    casualties_in_victories[i_cas] = pct_cas
            if casualties_in_victories:
                st.write("In successful battles:")
                for casualties, percentage in casualties_in_victories.items():
                    st.write(
                        f"- {percentage:.1f}% saw {casualties} {'player' if casualties == 1 else 'players'} fall"
                    )
            else:
                st.write(
                    "In successful battles, the party never suffered casualties (or no victories to analyze)."
                )
        else:
            st.write("Run simulation with victories to analyze party casualties.")

        # Combat Duration Statistics
        st.subheader("Combat Duration Statistics")
        if not results_df.empty:
            col1_dur, col2_dur, col3_dur = st.columns(3)
            with col1_dur:
                if not victory_df.empty:  # Check victory_df
                    st.write("Victory Round Distribution")
                    st.bar_chart(victory_df["rounds"].value_counts().sort_index())
                    st.metric(
                        "Median Victory Round", f"{victory_df['rounds'].median():.1f}"
                    )
                    st.metric(
                        "Mean Victory Round", f"{victory_df['rounds'].mean():.1f}"
                    )
                else:
                    st.write("No victories for round distribution.")
            with col2_dur:
                if not tpk_df.empty:  # Check tpk_df
                    st.write("TPK Round Distribution")
                    st.bar_chart(tpk_df["rounds"].value_counts().sort_index())
                    st.metric("Median TPK Round", f"{tpk_df['rounds'].median():.1f}")
                    st.metric("Mean TPK Round", f"{tpk_df['rounds'].mean():.1f}")
            with col3_dur:
                st.write("First Casualty Analysis")
                if "first_casualty_round" in results_df.columns:
                    casualty_data = results_df[
                        results_df["first_casualty_round"].notna()
                    ]
                    if not casualty_data.empty:
                        st.bar_chart(
                            casualty_data["first_casualty_round"]
                            .value_counts()
                            .sort_index()
                        )
                        st.metric(
                            "Avg Round of First Casualty",
                            f"{casualty_data['first_casualty_round'].mean():.1f}",
                        )
                        st.metric(
                            "Median Round of First Casualty",
                            f"{casualty_data['first_casualty_round'].median():.1f}",
                        )
                    else:
                        st.metric("First Casualty", "No casualties recorded")
                else:
                    st.metric("First Casualty", "Tracking not available or no data")
        else:
            st.write("Run simulation to see duration statistics.")

        # Encounter Balance Verdict
        st.subheader("Encounter Balance Analysis")
        if not victory_df.empty:  # Check victory_df
            victory_rounds = victory_df["rounds"]
            round_stats = {
                "min": victory_rounds.min(),
                "q1": victory_rounds.quantile(0.25),
                "median": victory_rounds.median(),
                "q3": victory_rounds.quantile(0.75),
                "max": victory_rounds.max(),
            }
            # Ensure victory_pct is defined if victory_df is not empty
            total_sims = len(results_df)  # Should be > 0 if victory_df is not empty
            victory_pct = (len(victory_df) / total_sims * 100) if total_sims > 0 else 0

            if victory_pct >= 80:
                verdict, color, details = (
                    "Easy Encounter",
                    "🟢",
                    "High probability of party victory",
                )
            elif victory_pct >= 60:
                verdict, color, details = (
                    "Moderate Challenge",
                    "🟡",
                    "Party favored but not guaranteed",
                )
            elif victory_pct >= 40:
                verdict, color, details = (
                    "Hard Challenge",
                    "🟠",
                    "Balanced but difficult",
                )
            elif victory_pct >= 20:
                verdict, color, details = (
                    "Very Hard",
                    "🔴",
                    "Party disadvantaged but victory possible",
                )
            else:
                verdict, color, details = (
                    "Extreme Challenge",
                    "⚫",
                    "Victory unlikely",
                )
            st.markdown(f"### {color} {verdict}")
            st.write(details)
            st.write("Combat Duration Analysis (Victories):")
            st.write(f"- Quickest Victory: Round {round_stats['min']}")
            st.write(
                f"- Typical Range: Round {round_stats['q1']:.1f} to {round_stats['q3']:.1f}"
            )
            st.write(f"- Longest Victory: Round {round_stats['max']}")
            if not tpk_df.empty:  # check tpk_df
                tpk_pct = (len(tpk_df) / total_sims * 100) if total_sims > 0 else 0
                st.write(
                    f"- Party Wipe Risk: {tpk_pct:.1f}% chance, typically around round {tpk_df['rounds'].median():.1f}"
                )
        elif not results_df.empty:  # No victories, but simulation was run
            st.markdown("### ⚫ Unwinnable")
            st.write("Party unable to achieve victory in any simulation")
            if not tpk_df.empty:  # check tpk_df
                st.write(f"Average TPK occurs on round {tpk_df['rounds'].mean():.1f}")
        else:  # results_df is empty (simulation not run)
            st.write("Run simulation to see balance analysis.")

        # Display sample combat log
        if (
            not results_df.empty
            and "combat_logs" in results_df.columns
            and results_df.loc[0, "combat_logs"]
        ):
            st.subheader("📜 Sample Combat Log")
            if "rng_seed_for_log" not in st.session_state:
                st.session_state.rng_seed_for_log = seed  # seed is from sidebar
            log_rng = np.random.default_rng(st.session_state.rng_seed_for_log)
            combat_log_list = results_df.loc[0, "combat_logs"]
            if combat_log_list:  # Should always be true if outer condition met
                sample_idx = log_rng.integers(0, len(combat_log_list))
                st.markdown(
                    '<div class="round-container combat-log">', unsafe_allow_html=True
                )
                st.markdown(combat_log_list[sample_idx])
                st.markdown("</div>", unsafe_allow_html=True)
            # else case not strictly needed due to outer check, but good for safety if structure changes
            # else:
            #     st.write("No combat logs available in this sample.")
        elif (
            not results_df.empty
        ):  # Sim run, but no logs (should not happen with current logic)
            st.write("No combat logs generated or available in this sample.")
        # If results_df is empty, nothing is displayed here, which is fine. Add message if desired:
        # else:
        # st.write("Run simulation to generate combat logs.")

        # Download button
        if not results_df.empty:
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Raw Simulation Data",
                csv,
                "encounter_simulation_results.csv",
                "text/csv",
                key="sim_download_button",
            )
        else:
            st.button(
                "Download Raw Simulation Data",
                disabled=True,
                help="Run simulation to generate data for download.",
                key="sim_download_button_disabled",  # Different key if needed
            )

    with tab2:  ########## ENCOUNTER BUILDER TAB ##########
        st.header("Build an Encounter to Meet Specific Outcomes")
        st.markdown("*(Work in Progress for iterative adjustments...)*")

        st.subheader("1. Set Desired Outcomes")
        if "builder_opt_goal" not in st.session_state:
            st.session_state.builder_opt_goal = "Target Victory %"
        st.selectbox(
            "Primary Optimization Goal:",
            ["Target Victory %", "Target First Casualty %"],
            key="builder_opt_goal",
        )
        optimization_goal = st.session_state.builder_opt_goal

        if "builder_target_perc" not in st.session_state:
            st.session_state.builder_target_perc = 80
        st.slider(
            f"Desired {optimization_goal.split('Target ')[1]}",
            min_value=0,
            max_value=100,
            step=5,
            key="builder_target_perc",
        )
        target_percentage = st.session_state.builder_target_perc

        st.subheader("2. Choose Starting Encounter Template")
        if "builder_template" not in st.session_state:
            st.session_state.builder_template = list(BUILDER_TEMPLATES.keys())[0]
        st.selectbox(
            "Select Encounter Template:",
            list(BUILDER_TEMPLATES.keys()),
            key="builder_template",
        )
        encounter_template_name = st.session_state.builder_template

        if st.button("Generate Encounter", key="builder_generate_button"):
            if not party:
                st.error("Please configure the party in the sidebar first.")
            else:
                # --- Initialize/clear session state for builder output ---
                st.session_state.builder_status_message = (
                    "Starting encounter generation..."
                )
                st.session_state.builder_iteration_log = [
                    "▶️ Generation process started."
                ]
                st.session_state.builder_final_encounter_summary = None
                st.session_state.builder_final_enemies_raw = None
                st.session_state.builder_final_sim_results = None
                # Clear old single-step results if they exist, to avoid confusion
                if "initial_sim_results" in st.session_state:
                    del st.session_state.initial_sim_results
                if "adjusted_sim_results" in st.session_state:
                    del st.session_state.adjusted_sim_results
                if "initial_template_summary" in st.session_state:
                    del st.session_state.initial_template_summary
                if "adjustment_description" in st.session_state:
                    del st.session_state.adjustment_description
                if "adjusted_enemies_summary" in st.session_state:
                    del st.session_state.adjusted_enemies_summary

                # --- Get current settings ---
                selected_template_info = BUILDER_TEMPLATES[encounter_template_name]
                template_enemy_defs_for_generation = selected_template_info["enemies"]
                # party_size is from sidebar
                # optimization_goal is from selectbox
                # target_percentage is from slider
                # seed is from sidebar

                # --- Generate initial enemies from template (similar to before) ---
                initial_generated_enemies_list = []
                initial_summary_list = []  # For the first log entry
                for enemy_def_for_gen in template_enemy_defs_for_generation:
                    count = enemy_def_for_gen["count_func"](party_size)
                    summary_entry = f"{count} x {enemy_def_for_gen['type']} (HP: {enemy_def_for_gen['hp']}, DR: {enemy_def_for_gen['dr']}, Dmg: {enemy_def_for_gen['damage_dice']})"
                    initial_summary_list.append(summary_entry)
                    for _ in range(count):
                        new_e = Enemy(
                            hp=enemy_def_for_gen["hp"],
                            dr=enemy_def_for_gen["dr"],
                            damage_dice=enemy_def_for_gen["damage_dice"],
                        )
                        new_e.type_for_template = enemy_def_for_gen["type"]
                        initial_generated_enemies_list.append(new_e)

                st.session_state.builder_iteration_log.append(
                    f"Selected Template: '{encounter_template_name}'"
                )
                st.session_state.builder_iteration_log.append(
                    f"Initial Encounter based on template: {', '.join(initial_summary_list)}"
                )

                if not initial_generated_enemies_list:
                    st.warning(
                        "Initial template generated no enemies. Builder stopped."
                    )
                    st.session_state.builder_status_message = (
                        "Initial template resulted in no enemies."
                    )
                else:
                    # --- Builder Loop Parameters ---
                    max_builder_iterations = 20
                    sim_iters_for_iteration = 100
                    sim_iters_for_final = 10000
                    encounter_found_within_tolerance = False

                    # --- Simple Sequential Rule Application ---
                    current_enemies = copy.deepcopy(initial_generated_enemies_list)

                    # Check if initial template is already in tolerance
                    rng_initial_check = np.random.default_rng(seed)
                    initial_results_df = simulate_encounter(
                        party,
                        current_enemies,
                        sim_iters_for_iteration,
                        rng_initial_check,
                    )
                    initial_metric = get_metric_from_df(
                        initial_results_df, optimization_goal
                    )
                    st.session_state.builder_iteration_log.append(
                        f"Initial template metric: {initial_metric:.1f}% (Target: {target_percentage}%)"
                    )

                    if (
                        abs(initial_metric - target_percentage)
                        <= TARGET_METRIC_TOLERANCE
                    ):
                        st.session_state.builder_iteration_log.append(
                            f"✅ Initial template already within tolerance! Finalizing."
                        )
                        encounter_found_within_tolerance = True
                    else:
                        # Determine direction needed
                        if optimization_goal == "Target Victory %":
                            need_harder = initial_metric > target_percentage
                        elif optimization_goal == "Target First Casualty %":
                            need_harder = initial_metric < target_percentage

                        adjustment_direction = 1 if need_harder else -1
                        direction_desc = "harder" if need_harder else "easier"
                        st.session_state.builder_iteration_log.append(
                            f"Need to make encounter {direction_desc} (adjustment direction: {adjustment_direction})"
                        )

                        # Apply each rule sequentially
                        adjustment_rules = selected_template_info["adjustment_priority"]
                        for rule_idx, rule in enumerate(adjustment_rules):
                            st.session_state.builder_iteration_log.append(
                                f"\n📐 Rule {rule_idx + 1}/{len(adjustment_rules)}: {rule['param']} for {rule['target_enemy_type']}"
                            )
                            st.session_state.builder_iteration_log.append(
                                f"  Starting with: {'; '.join(summarize_enemies(current_enemies))}"
                            )

                            # Resolve max_increase if it's a lambda
                            max_increase_val = rule["max_increase"]
                            if callable(max_increase_val):
                                max_increase_val = max_increase_val(party_size)

                            # Apply this rule step by step until max or target reached
                            step_size = rule["step"]
                            max_steps = (
                                max_increase_val // step_size if step_size > 0 else 0
                            )

                            # Start with current state and push this rule to maximum
                            rule_start_enemies = copy.deepcopy(current_enemies)

                            for step_num in range(1, max_steps + 1):
                                # Apply this step (cumulative from rule start, not overall start)
                                test_enemies, step_desc = apply_single_adjustment(
                                    rule_start_enemies,  # Start from beginning of this rule
                                    template_enemy_defs_for_generation,
                                    rule,
                                    party_size,
                                    adjustment_direction
                                    * step_num,  # Cumulative steps for this rule
                                )

                                # Simulate the test
                                rng_test = np.random.default_rng(seed)
                                test_results_df = simulate_encounter(
                                    party,
                                    test_enemies,
                                    sim_iters_for_iteration,
                                    rng_test,
                                )
                                test_metric = get_metric_from_df(
                                    test_results_df, optimization_goal
                                )

                                st.session_state.builder_iteration_log.append(
                                    f"  Step {step_num}/{max_steps}: {len(test_enemies)} enemies, metric: {test_metric:.1f}%"
                                )

                                # Always adopt this step (we're pushing to maximum)
                                current_enemies = copy.deepcopy(test_enemies)

                                # Check if we're in tolerance
                                if (
                                    abs(test_metric - target_percentage)
                                    <= TARGET_METRIC_TOLERANCE
                                ):
                                    st.session_state.builder_iteration_log.append(
                                        f"  🎯 Target reached! Finalizing with this configuration."
                                    )
                                    encounter_found_within_tolerance = True
                                    break

                            if encounter_found_within_tolerance:
                                break  # Exit rule loop

                            # Log the final state after this rule
                            st.session_state.builder_iteration_log.append(
                                f"  ✅ Rule {rule_idx + 1} completed. New state: {'; '.join(summarize_enemies(current_enemies))}"
                            )

                    # --- Final simulation and results ---
                    if encounter_found_within_tolerance:
                        st.session_state.builder_status_message = "Encounter within tolerance found. Running final simulation..."
                    else:
                        st.session_state.builder_status_message = "Could not reach target with available rules. Using best attempt..."

                    # Final high-iteration simulation
                    final_rng = np.random.default_rng(seed)
                    final_results_df = simulate_encounter(
                        party, current_enemies, sim_iters_for_final, final_rng
                    )
                    st.session_state.builder_final_sim_results = final_results_df
                    st.session_state.builder_final_enemies_raw = copy.deepcopy(
                        current_enemies
                    )
                    st.session_state.builder_final_encounter_summary = (
                        summarize_enemies(current_enemies, detail=False)
                    )  # Use compact summary

                    final_metric_value = get_metric_from_df(
                        final_results_df, optimization_goal
                    )
                    st.session_state.builder_iteration_log.append(
                        f"\n🎯 Final simulation ({sim_iters_for_final} iterations): {final_metric_value:.1f}%"
                    )

                    # --- Auto-populate sidebar enemy configuration ---
                    def store_suggested_encounter(enemies_list):
                        """Store suggested encounter for later application to sidebar."""
                        # Map template types to sidebar indices
                        type_to_index = {
                            "Minion": 0,
                            "Soldier": 1,
                            "Elite": 2,
                            "Boss": 3,
                        }

                        # Group enemies by type and stats
                        enemy_groups = {}
                        for enemy in enemies_list:
                            enemy_type = getattr(enemy, "type_for_template", "Minion")
                            key = (enemy_type, enemy.hp, enemy.dr, enemy.damage_dice)
                            if key not in enemy_groups:
                                enemy_groups[key] = 0
                            enemy_groups[key] += 1

                        # Store suggested configuration
                        suggested_config = {}
                        for i in range(4):  # Initialize all to 0
                            suggested_config[f"count_{i}"] = 0
                            suggested_config[f"hp_{i}"] = 5  # Default values
                            suggested_config[f"dr_{i}"] = 0
                            suggested_config[f"dice_{i}"] = "1d4"

                        # Fill in the actual suggested values
                        for (
                            enemy_type,
                            hp,
                            dr,
                            damage_dice,
                        ), count in enemy_groups.items():
                            if enemy_type in type_to_index:
                                idx = type_to_index[enemy_type]
                                suggested_config[f"count_{idx}"] = count
                                suggested_config[f"hp_{idx}"] = hp
                                suggested_config[f"dr_{idx}"] = dr
                                suggested_config[f"dice_{idx}"] = damage_dice

                        st.session_state.builder_suggested_config = suggested_config
                        st.session_state.builder_iteration_log.append(
                            f"📋 Suggested encounter configuration stored. Use 'Apply to Simulator' button to load it."
                        )

                    store_suggested_encounter(current_enemies)

                    if encounter_found_within_tolerance:
                        st.session_state.builder_status_message = f"🎉 Encounter generated! Final Metric ({optimization_goal.split('Target ')[1]}): {final_metric_value:.1f}%"
                    else:
                        st.session_state.builder_status_message = f"⚠️ Best attempt: {final_metric_value:.1f}% (Target: {target_percentage}%)"

        # --- Display Area for Builder (Modified) ---
        if "builder_status_message" in st.session_state:
            st.info(st.session_state.builder_status_message)

        if (
            "builder_iteration_log" in st.session_state
            and st.session_state.builder_iteration_log
        ):
            with st.expander("Show Iteration Log", expanded=False):
                for log_entry in st.session_state.builder_iteration_log:
                    st.text(
                        log_entry
                    )  # Using st.text for now, can be st.markdown for more formatting

        if (
            "builder_final_encounter_summary" in st.session_state
            and st.session_state.builder_final_encounter_summary
        ):
            st.markdown("--- ")
            st.markdown("### Suggested Encounter")
            for item in st.session_state.builder_final_encounter_summary:
                st.markdown(f"- {item}")

            # Add Apply to Simulator button
            if "builder_suggested_config" in st.session_state:
                col1, col2 = st.columns([1, 2])
                with col1:
                    if st.button(
                        "🔄 Apply to Simulator",
                        help="Load this encounter into the sidebar for simulation",
                    ):
                        # Store config as pending (to be applied on next page load)
                        st.session_state.builder_pending_config = (
                            st.session_state.builder_suggested_config.copy()
                        )
                        st.rerun()
                with col2:
                    st.info(
                        "💡 Click to load this encounter into the Combat Simulator tab's enemy configuration."
                    )

            if (
                "builder_final_sim_results" in st.session_state
                and st.session_state.builder_final_sim_results is not None
            ):
                results_df_final_display = st.session_state.builder_final_sim_results
                final_vic_pct = get_metric_from_df(
                    results_df_final_display, "Target Victory %"
                )
                final_fc_pct = get_metric_from_df(
                    results_df_final_display, "Target First Casualty %"
                )
                st.metric(
                    f"Expected Victory % (after {len(results_df_final_display)} sims)",
                    f"{final_vic_pct:.1f}%",
                )
                st.metric(
                    f"Expected First Casualty % (after {len(results_df_final_display)} sims)",
                    f"{final_fc_pct:.1f}%",
                )
            # If no builder results yet, just show nothing instead of placeholder JSON

        st.markdown("---")
        # Removed the old static JSON placeholder, as the logic above will show something more relevant


if __name__ == "__main__":
    main()
