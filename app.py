"""
Healthcare Chatbot Web Application
==================================
Flask-based disease prediction system using a trained Random Forest model.

This application:
1. Loads a pre-trained disease classification model
2. Accepts user input through a web form
3. Preprocesses input to match training data structure
4. Predicts disease and displays results

Author: Healthcare AI Team
Version: 1.0
"""

# ============================================
# IMPORTS
# ============================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================
# FLASK APP INITIALIZATION
# ============================================

app = Flask(__name__, template_folder=BASE_DIR)

# ============================================
# LOAD MODEL AND ENCODER
# ============================================

def load_env_file(env_path):
    """Load simple KEY=VALUE pairs from a .env file into process environment."""
    if not os.path.exists(env_path):
        return

    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(os.path.join(BASE_DIR, '.env'))

# Load the trained disease prediction model
MODEL_PATH = os.path.join(BASE_DIR, 'disease_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Grok (xAI) API configuration
GROK_API_URL = os.getenv('GROK_API_URL', 'https://api.x.ai/v1/chat/completions')
GROK_MODEL = os.getenv('GROK_MODEL', 'grok-3-mini')
GROK_API_KEY = os.getenv('GROK_API_KEY', '')

model = None
label_encoder = None

def load_model():
    """Load model with memory error handling."""
    global model, label_encoder
    if model is not None:
        return True
    try:
        print("Loading model... (this may take 1-2 minutes for large models)")
        label_encoder = joblib.load(ENCODER_PATH)
        print("✓ Label encoder loaded!")
        model = joblib.load(MODEL_PATH)
        print("✓ Model loaded successfully!")
        return True
    except FileNotFoundError as e:
        print(f"✗ Error loading model files: {e}")
        print("Please ensure disease_model.pkl and label_encoder.pkl are in the same directory as app.py")
        return False
    except MemoryError:
        print("✗ MEMORY ERROR: Not enough RAM to load the model.")
        print("  Solutions:")
        print("  1. Close other applications (browsers, VS Code, etc.)")
        print("  2. Restart your computer and try again")
        print("  3. Your system needs at least 8GB free RAM for this model")
        return False
    except Exception as e:
        if "MemoryError" in str(type(e).__name__) or "memory" in str(e).lower():
            print("✗ MEMORY ERROR: Not enough RAM to load the model.")
            print("  Close other applications and try again.")
        else:
            print(f"✗ Error loading model: {e}")
        return False

# Try to load on startup
load_model()


def _extract_json_block(raw_text):
    """Extract JSON object from model output, including fenced markdown content."""
    if not raw_text:
        return None

    text = raw_text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1]).strip()

    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return None
    return text[start_idx:end_idx + 1]


def build_default_guidance(predicted_disease):
    """Fallback guidance if Grok is unavailable."""
    return {
        'summary': f"Possible condition pattern detected: {predicted_disease}.",
        'precautions': [
            'Rest and stay hydrated.',
            'Avoid self-medicating with strong prescription medicines.',
            'Monitor symptom severity, frequency, and triggers.'
        ],
        'next_steps': [
            'Book a consultation with a licensed healthcare professional.',
            'Share your symptom history and this assessment result during the visit.',
            'Follow clinician advice and prescribed investigations.'
        ],
        'urgent_red_flags': [
            'Severe chest pain, breathing difficulty, fainting, confusion, or uncontrolled bleeding.',
            'If symptoms rapidly worsen, seek urgent care immediately.'
        ],
        'disclaimer': 'AI guidance is informational only and not a diagnosis or treatment plan.'
    }


def generate_grok_guidance(top_predictions, selected_symptoms, form_data):
    """Generate disease precautions and actionable steps using Grok."""
    if not top_predictions:
        return build_default_guidance('Unknown condition')

    predicted_disease = top_predictions[0]['disease']

    if not GROK_API_KEY:
        return build_default_guidance(predicted_disease)

    prediction_lines = [
        f"- {pred['disease']} ({pred['confidence']}%, {pred['level']})"
        for pred in top_predictions
    ]
    symptoms_text = ', '.join(selected_symptoms) if selected_symptoms else 'No specific symptoms selected'

    user_prompt = (
        'You are helping with symptom-triage guidance for educational use.\n'
        'Given a machine-learning disease prediction and symptoms, provide practical precautions and next steps.\n'
        'Do not provide medication dosage.\n'
        'Always include emergency red flags.\n'
        'Respond ONLY in valid JSON with this schema:\n'
        '{\n'
        '  "summary": "string",\n'
        '  "precautions": ["string", "..."],\n'
        '  "next_steps": ["string", "..."],\n'
        '  "urgent_red_flags": ["string", "..."],\n'
        '  "disclaimer": "string"\n'
        '}\n\n'
        f"Top predictions:\n{chr(10).join(prediction_lines)}\n\n"
        f"Selected symptoms: {symptoms_text}\n"
        f"Age: {form_data.get('age', 'N/A')}\n"
        f"Gender: {form_data.get('gender', 'N/A')}\n"
        f"BMI: {form_data.get('bmi', 'N/A')}\n"
        f"Smoking status: {form_data.get('smoking_status', 'N/A')}"
    )

    payload = {
        'model': GROK_MODEL,
        'temperature': 0.2,
        'max_tokens': 700,
        'messages': [
            {
                'role': 'system',
                'content': 'Provide safe, concise health guidance in strict JSON. Do not add markdown wrappers.'
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
    }

    request = Request(
        GROK_API_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROK_API_KEY}'
        },
        method='POST'
    )

    try:
        with urlopen(request, timeout=20) as response:
            response_body = response.read().decode('utf-8')

        body_json = json.loads(response_body)
        content = body_json['choices'][0]['message']['content']
        json_block = _extract_json_block(content)
        if not json_block:
            return build_default_guidance(predicted_disease)

        parsed = json.loads(json_block)

        # Normalize shape to avoid template/API breaks.
        return {
            'summary': parsed.get('summary', f'Possible condition pattern detected: {predicted_disease}.'),
            'precautions': parsed.get('precautions', []),
            'next_steps': parsed.get('next_steps', []),
            'urgent_red_flags': parsed.get('urgent_red_flags', []),
            'disclaimer': parsed.get('disclaimer', 'AI guidance is informational only and not a diagnosis or treatment plan.')
        }
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError):
        return build_default_guidance(predicted_disease)

# ============================================
# DEFINE TRAINING COLUMN STRUCTURE
# ============================================

# These are ALL the feature columns used during training (after one-hot encoding)
# The model expects exactly 832 columns in this specific order

# List of all symptom severity columns from training data
SYMPTOM_COLUMNS = [
    'IBS_symptoms_severity', 'abdominal_bloating_severity', 'abdominal_cramps_severity',
    'abdominal_discomfort_severity', 'abdominal_distension_severity', 'abdominal_fullness_severity',
    'abdominal_obesity_severity', 'abdominal_pain_severity', 'abdominal_pain_left_severity',
    'abdominal_pain_right_severity', 'abdominal_swelling_severity', 'abdominal_tenderness_severity',
    'abnormal_cells_severity', 'abnormal_eating_patterns_severity', 'abnormal_skull_shape_severity',
    'acid_reflux_severity', 'acid_taste_severity', 'acne_severity', 'anal_itching_severity',
    'anemia_severity', 'ankle_swelling_severity', 'anxiety_severity', 'anxiety_leaving_home_severity',
    'appetite_changes_severity', 'arch_absence_severity', 'arm_pain_severity', 'arm_weakness_severity',
    'asking_repetition_severity', 'asymmetric_mole_severity', 'avoidance_behavior_severity',
    'avoidance_behaviors_severity', 'back_pain_severity', 'back_stiffness_severity', 'bad_breath_severity',
    'bad_taste_mouth_severity', 'balance_problems_severity', 'bald_patches_severity', 'belching_severity',
    'bent_toe_severity', 'binge_eating_severity', 'black_dots_severity', 'blackheads_severity',
    'bladder_pain_severity', 'bladder_problems_severity', 'blanching_severity', 'bleeding_severity',
    'bleeding_between_periods_severity', 'bleeding_gums_severity', 'bleeding_lesion_severity',
    'blistering_severity', 'blisters_severity', 'blisters_lip_severity', 'blisters_mouth_severity',
    'bloating_severity', 'blood_in_stool_severity', 'blood_in_urine_severity', 'blood_in_urine_stool_severity',
    'bloody_diarrhea_severity', 'bloody_stool_severity', 'blurred_central_vision_severity',
    'body_ache_severity', 'body_image_issues_severity', 'body_rash_severity', 'bone_deformities_severity',
    'bone_fracture_severity', 'bone_pain_severity', 'bone_spurs_severity', 'bowel_habit_changes_severity',
    'brain_fog_severity', 'breast_lump_severity', 'breast_pain_severity', 'breast_swelling_severity',
    'breathing_stops_severity', 'brittle_nails_severity', 'bruising_severity', 'buffalo_hump_severity',
    'bulge_behind_knee_severity', 'bulls_eye_rash_severity', 'burning_severity', 'burning_chest_severity',
    'burning_eyes_severity', 'burning_pain_severity', 'burning_sensation_severity', 'burning_skin_severity',
    'burning_stomach_pain_severity', 'burrow_tracks_severity', 'butterfly_rash_severity',
    'buzzing_sound_severity', 'calf_muscle_underdevelopment_severity', 'calf_pain_severity',
    'cataplexy_severity', 'changing_mole_severity', 'chest_discomfort_severity', 'chest_pain_severity',
    'chest_tightness_severity', 'chest_wall_pain_severity', 'chills_severity', 'chills_after_severity',
    'choking_severity', 'chronic_cough_severity', 'chronic_emptiness_severity', 'clammy_skin_severity',
    'clay_colored_stool_severity', 'cloudy_urine_severity', 'cloudy_vision_severity',
    'clubbing_fingers_severity', 'coated_tongue_severity', 'cobweb_vision_severity',
    'cobwebs_vision_severity', 'cognitive_problems_severity', 'cold_hands_feet_severity',
    'cold_intolerance_severity', 'cold_sensitivity_severity', 'coldness_leg_foot_severity',
    'color_variation_mole_severity', 'colors_appear_faded_severity', 'compulsive_behavior_severity',
    'compulsive_behaviors_severity', 'concentration_problems_severity', 'confusion_severity',
    'congestion_severity', 'conjunctivitis_severity', 'constipation_severity', 'coordination_issues_severity',
    'coordination_problems_severity', 'corn_callus_severity', 'cough_severity', 'cough_night_severity',
    'cough_with_mucus_severity', 'coughing_blood_severity', 'coughing_eating_severity',
    'cracked_skin_severity', 'cracking_severity', 'cracking_mouth_corners_severity', 'cramps_severity',
    'craving_severity', 'craving_alcohol_severity', 'cravings_severity', 'crusting_eyelids_severity',
    'crusty_eyelashes_severity', 'crusty_eyelid_severity', 'curtain_over_vision_severity',
    'cyanosis_severity', 'dark_areas_vision_severity', 'dark_spot_severity', 'dark_spots_vision_severity',
    'dark_urine_severity', 'daytime_sleepiness_severity', 'decreased_motion_severity',
    'decreased_urine_severity', 'deep_pain_severity', 'deformity_severity', 'dehydration_severity',
    'delayed_growth_severity', 'delusions_severity', 'dental_problems_severity', 'depressed_mood_severity',
    'depression_severity', 'developmental_delays_severity', 'diarrhea_severity',
    'difficulty_breathing_severity', 'difficulty_chewing_severity', 'difficulty_closing_eye_severity',
    'difficulty_communicating_severity', 'difficulty_conceiving_severity', 'difficulty_eating_severity',
    'difficulty_erection_severity', 'difficulty_falling_asleep_severity', 'difficulty_gripping_severity',
    'difficulty_lifting_severity', 'difficulty_motor_planning_severity', 'difficulty_opening_mouth_severity',
    'difficulty_organizing_severity', 'difficulty_phone_conversations_severity',
    'difficulty_physical_tasks_severity', 'difficulty_reaching_severity', 'difficulty_reading_severity',
    'difficulty_rhyming_severity', 'difficulty_rising_severity', 'difficulty_standing_severity',
    'difficulty_starting_urination_severity', 'difficulty_staying_asleep_severity',
    'difficulty_straightening_fingers_severity', 'difficulty_swallowing_severity',
    'difficulty_waiting_severity', 'difficulty_waking_severity', 'difficulty_walking_severity',
    'discharge_severity', 'discoloration_severity', 'discomfort_severity', 'discomfort_contacts_severity',
    'discomfort_sitting_severity', 'disorganized_thinking_severity', 'disorientation_severity',
    'disrupted_night_sleep_severity', 'distinctive_facial_features_severity', 'distorted_vision_severity',
    'distress_severity', 'dizziness_severity', 'double_vision_severity', 'dribbling_severity',
    'drinking_despite_problems_severity', 'drooling_severity', 'drooping_eyelid_severity',
    'drooping_face_severity', 'drowsiness_severity', 'dry_cough_severity', 'dry_cracking_skin_severity',
    'dry_eyes_severity', 'dry_mouth_severity', 'dry_scaly_skin_severity', 'dry_skin_severity',
    'ear_drainage_severity', 'ear_infections_severity', 'ear_pain_severity', 'early_fullness_severity',
    'early_waking_severity', 'easy_bleeding_severity', 'easy_bruising_severity', 'elbow_pain_severity',
    'elevated_mood_severity', 'emotional_flatness_severity', 'emotional_instability_severity',
    'emotional_numbness_severity', 'empty_scrotum_severity', 'enlarged_calves_severity',
    'enlarged_liver_severity', 'erectile_dysfunction_severity', 'evolving_mole_severity',
    'excess_hair_growth_severity', 'excess_weight_severity', 'excessive_daytime_sleepiness_severity',
    'excessive_tearing_severity', 'excessive_thirst_severity', 'exclamation_point_hairs_severity',
    'exhaustion_severity', 'extreme_hunger_severity', 'eye_bruising_severity', 'eye_burning_severity',
    'eye_color_change_severity', 'eye_contact_avoidance_severity', 'eye_discharge_severity',
    'eye_dryness_severity', 'eye_fatigue_severity', 'eye_inflammation_severity', 'eye_irritation_severity',
    'eye_pain_severity', 'eye_problems_severity', 'eye_redness_severity', 'eye_watering_severity',
    'eyelid_bump_severity', 'eyelid_inflammation_severity', 'eyelid_pain_severity',
    'facial_bumps_severity', 'facial_drooping_severity', 'facial_numbness_severity', 'facial_pain_severity',
    'facial_pressure_severity', 'facial_rash_severity', 'facial_redness_severity', 'facial_swelling_severity',
    'facial_weakness_severity', 'faded_colors_severity', 'fainting_severity', 'fatigue_severity',
    'fear_contamination_severity', 'fear_helplessness_severity', 'fear_of_abandonment_severity',
    'fear_of_dying_severity', 'fear_open_spaces_severity', 'feeding_difficulties_severity',
    'feeling_full_quickly_severity', 'feeling_heartbeat_severity', 'feelings_of_worthlessness_severity',
    'fever_severity', 'finger_contracture_severity', 'finger_numbness_severity', 'finger_pain_severity',
    'finger_swelling_severity', 'flaking_severity', 'flapping_tremor_severity', 'flashbacks_severity',
    'flesh_colored_bumps_severity', 'floaters_severity', 'floating_spots_severity',
    'flu_like_symptoms_severity', 'fluid_drainage_severity', 'flushing_severity', 'foamy_urine_severity',
    'foot_itching_severity', 'foot_pain_severity', 'foot_pain_morning_severity', 'foot_stiffness_severity',
    'forgetfulness_severity', 'frequent_bowel_severity', 'frequent_falls_severity',
    'frequent_infections_severity', 'frequent_lung_infections_severity',
    'frequent_prescription_changes_severity', 'frequent_urination_severity', 'fussiness_severity',
    'gas_severity', 'gasping_for_air_severity', 'genital_discharge_severity', 'genital_sores_severity',
    'genital_warts_severity', 'glare_sensitivity_severity', 'gray_membrane_throat_severity',
    'gritty_feeling_severity', 'groin_pain_severity', 'growth_problems_severity', 'gum_pockets_severity',
    'hair_loss_severity', 'hair_loss_legs_severity', 'hair_thinning_severity', 'hallucinations_severity',
    'halos_around_lights_severity', 'hand_numbness_tingling_severity', 'hard_stool_severity',
    'head_pressure_severity', 'head_tremor_severity', 'headache_severity', 'hearing_loss_severity',
    'heart_defects_severity', 'heart_murmur_severity', 'heartburn_severity', 'heat_intolerance_severity',
    'heaviness_severity', 'heaviness_groin_severity', 'heavy_bleeding_severity', 'heel_pain_severity',
    'hiccups_severity', 'high_blood_pressure_severity', 'high_blood_sugar_severity',
    'high_body_temp_severity', 'hip_pain_severity', 'hives_severity', 'hoarseness_severity',
    'honey_colored_crusts_severity', 'hunger_severity', 'hydrophobia_severity', 'hyperactivity_severity',
    'hypervigilance_severity', 'identity_disturbance_severity', 'impulsive_behavior_severity',
    'impulsivity_severity', 'inability_to_move_severity', 'inattention_severity',
    'incomplete_emptying_severity', 'incomplete_evacuation_severity', 'inconsistent_errors_severity',
    'increased_appetite_severity', 'increased_energy_severity', 'increased_head_pressure_severity',
    'increased_hunger_severity', 'increased_saliva_severity', 'increased_shedding_severity',
    'indigestion_severity', 'infection_signs_severity', 'infertility_severity', 'inflammation_severity',
    'infrequent_bowel_severity', 'inguinal_swelling_severity', 'inner_elbow_pain_severity',
    'instability_severity', 'intellectual_disability_severity', 'intense_itching_severity',
    'intrusive_thoughts_severity', 'involuntary_movements_severity', 'irregular_border_mole_severity',
    'irregular_bowel_severity', 'irregular_heartbeat_severity', 'irregular_periods_severity',
    'irritability_severity', 'irritated_skin_severity', 'itching_severity', 'itchy_blisters_severity',
    'itchy_eyelids_severity', 'itchy_eyes_severity', 'itchy_rash_severity', 'itchy_skin_severity',
    'itchy_throat_severity', 'jaundice_severity', 'jaw_clicking_severity', 'jaw_pain_severity',
    'jaw_stiffness_severity', 'joint_clicking_severity', 'joint_pain_severity', 'joint_pain_swelling_severity',
    'joint_redness_severity', 'joint_stiffness_severity', 'joint_swelling_severity', 'joint_warmth_severity',
    'kidney_problems_severity', 'kidney_stones_severity', 'knee_pain_severity', 'knee_stiffness_severity',
    'knee_swelling_severity', 'lack_of_motivation_severity', 'lack_of_strength_severity',
    'lack_of_sweating_severity', 'large_diameter_mole_severity', 'learning_difficulties_severity',
    'left_upper_abdominal_pain_severity', 'leg_cramps_severity', 'leg_numbness_severity',
    'leg_pain_severity', 'leg_pain_walking_severity', 'leg_swelling_severity', 'leg_tingling_severity',
    'leg_weakness_severity', 'letter_confusion_severity', 'light_flashes_severity',
    'light_sensitivity_severity', 'limb_pain_severity', 'limb_swelling_severity',
    'limited_ankle_motion_severity', 'limited_arm_movement_severity', 'limited_flexibility_severity',
    'limited_mobility_severity', 'limited_motion_severity', 'limited_movement_severity',
    'limited_range_motion_severity', 'limited_shoulder_motion_severity', 'limping_severity',
    'lip_tingling_severity', 'localized_pain_severity', 'location_near_joint_severity',
    'loose_stool_severity', 'loose_teeth_severity', 'loss_of_appetite_severity',
    'loss_of_bowel_control_severity', 'loss_of_color_mouth_severity', 'loss_of_consciousness_severity',
    'loss_of_control_severity', 'loss_of_height_severity', 'loss_of_interest_severity',
    'loss_of_smell_severity', 'loss_of_taste_severity', 'loud_snoring_severity',
    'low_blood_pressure_severity', 'low_muscle_tone_severity', 'low_self_esteem_severity',
    'lower_back_pain_severity', 'lump_movement_severity', 'lump_near_anus_severity', 'malaise_severity',
    'memory_loss_severity', 'menstrual_cramps_severity', 'mild_swelling_severity', 'mood_changes_severity',
    'moon_face_severity', 'morning_headache_severity', 'mouth_sores_severity', 'movable_lump_severity',
    'movement_problems_severity', 'mucus_in_stool_severity', 'mucus_production_severity',
    'muffled_sounds_severity', 'muffled_voice_severity', 'muscle_ache_severity', 'muscle_cramps_severity',
    'muscle_jerking_severity', 'muscle_rigidity_severity', 'muscle_spasms_severity',
    'muscle_spasticity_severity', 'muscle_tension_severity', 'muscle_twitching_severity',
    'muscle_weakness_severity', 'nail_changes_severity', 'nail_damage_severity', 'nasal_congestion_severity',
    'nausea_severity', 'nausea_morning_severity', 'neck_lump_severity', 'neck_pain_severity',
    'neck_stiffness_severity', 'neck_swelling_severity', 'need_for_symmetry_severity',
    'neglecting_activities_severity', 'neglecting_responsibilities_severity', 'new_diabetes_severity',
    'new_skin_growth_severity', 'night_pain_severity', 'nightmares_severity', 'nighttime_itching_severity',
    'nipple_discharge_severity', 'nipple_retraction_severity', 'no_breathing_severity', 'no_pulse_severity',
    'nocturia_severity', 'non_healing_sore_severity', 'nosebleed_severity', 'nosebleeds_severity',
    'numbness_severity', 'numbness_feet_severity', 'numbness_tingling_severity', 'nystagmus_severity',
    'obsessive_thoughts_severity', 'oily_scalp_severity', 'oily_skin_severity', 'oozing_severity',
    'open_ulcer_severity', 'oral_thrush_severity', 'pain_severity', 'pain_after_standing_severity',
    'pain_at_bulge_severity', 'pain_at_site_severity', 'pain_behind_eyes_severity', 'pain_biting_severity',
    'pain_during_bowel_severity', 'pain_during_intercourse_severity', 'pain_episodes_severity',
    'pain_in_morning_severity', 'pain_lifting_severity', 'pain_or_discomfort_severity',
    'pain_tenderness_severity', 'pain_with_activity_severity', 'pain_with_breathing_severity',
    'pain_with_coughing_severity', 'pain_with_lifting_severity', 'pain_with_movement_severity',
    'pain_with_wrist_movement_severity', 'painful_bumps_severity', 'painful_chewing_severity',
    'painful_cramps_severity', 'painful_ejaculation_severity', 'painful_intercourse_severity',
    'painful_periods_severity', 'painful_rash_severity', 'painful_sores_mouth_severity',
    'painful_urination_severity', 'painless_severity', 'painless_lump_severity',
    'painless_swollen_lymph_nodes_severity', 'pale_skin_severity', 'pallor_severity',
    'palm_nodules_severity', 'palm_thickening_severity', 'palpitations_severity', 'panic_symptoms_severity',
    'paralysis_severity', 'patchy_hair_loss_severity', 'peeling_severity', 'pelvic_pain_severity',
    'peripheral_vision_loss_severity', 'personality_changes_severity', 'photosensitivity_severity',
    'physical_anxiety_symptoms_severity', 'pimple_like_rash_severity', 'pimples_severity',
    'pink_urine_severity', 'poor_feeding_severity', 'poor_night_vision_severity',
    'poor_weight_gain_severity', 'popping_sensation_severity', 'post_exertional_malaise_severity',
    'postnasal_drip_severity', 'premature_hair_graying_severity', 'profuse_diarrhea_severity',
    'prolonged_bleeding_severity', 'psychomotor_changes_severity', 'pulsating_sensation_severity',
    'purging_severity', 'purple_flat_bumps_severity', 'pus_severity', 'pus_drainage_severity',
    'racing_heart_severity', 'racing_thoughts_severity', 'radiating_leg_pain_severity',
    'radiating_pain_severity', 'raised_edges_severity', 'raised_scar_severity', 'raised_welts_severity',
    'rapid_breathing_severity', 'rapid_heartbeat_severity', 'rapid_weight_gain_severity', 'rash_severity',
    'rash_on_elbows_knees_severity', 'raynaud_phenomenon_severity', 'reading_difficulty_severity',
    'receding_gums_severity', 'receding_hairline_severity', 'rectal_bleeding_severity',
    'rectal_pain_severity', 'red_bump_severity', 'red_eyes_severity', 'red_gums_severity',
    'red_skin_severity', 'red_skin_diaper_area_severity', 'red_sores_severity', 'red_streaks_severity',
    'red_swollen_area_severity', 'redness_severity', 'redness_mouth_severity', 'reduced_alertness_severity',
    'reduced_desire_severity', 'reduced_exercise_capacity_severity', 'reduced_flexibility_severity',
    'reduced_smell_severity', 'reduced_taste_severity', 'regurgitation_severity',
    'relationship_problems_severity', 'repetitive_behaviors_severity', 'restlessness_severity',
    'restricted_interests_severity', 'restricted_motion_severity', 'restricted_toe_motion_severity',
    'right_upper_pain_severity', 'ring_shaped_rash_severity', 'ringing_ear_severity',
    'ringing_ears_severity', 'risky_behavior_severity', 'roaring_sound_severity', 'rose_spots_severity',
    'rough_skin_severity', 'rough_texture_severity', 'routine_dependence_severity', 'runny_nose_severity',
    'salt_craving_severity', 'salty_tasting_skin_severity', 'scaling_severity', 'scaling_skin_severity',
    'scalp_flaking_severity', 'scalp_itching_severity', 'scalp_tenderness_severity', 'scarring_severity',
    'scrotal_swelling_severity', 'secondary_infection_severity', 'seeing_double_severity',
    'seizure_with_fever_severity', 'seizures_severity', 'self_harm_severity', 'sensitivity_severity',
    'sensitivity_hot_cold_severity', 'sensitivity_to_touch_severity', 'sensory_sensitivities_severity',
    'severe_joint_pain_severity', 'severe_rash_severity', 'severe_sore_throat_severity',
    'severe_toothache_severity', 'shakiness_severity', 'shallow_breathing_severity', 'shivering_severity',
    'shooting_pain_severity', 'short_stature_severity', 'shortened_leg_severity',
    'shortness_of_breath_severity', 'shoulder_pain_severity', 'shoulder_stiffness_severity',
    'shoulder_tension_severity', 'silvery_scales_severity', 'skin_abscess_severity',
    'skin_changes_breast_severity', 'skin_color_changes_severity', 'skin_cracking_severity',
    'skin_darkening_severity', 'skin_discoloration_severity', 'skin_erosion_severity',
    'skin_growths_severity', 'skin_hardening_severity', 'skin_lesions_severity', 'skin_pain_severity',
    'skin_peeling_severity', 'skin_rash_severity', 'skin_rash_hands_feet_severity',
    'skin_sensitivity_severity', 'skin_sore_severity', 'skin_swelling_severity', 'skin_thickening_severity',
    'skin_thinning_severity', 'skin_tightness_severity', 'skin_warm_touch_severity', 'skin_warmth_severity',
    'skipped_beats_severity', 'slapped_cheek_appearance_severity', 'sleep_apnea_severity',
    'sleep_disturbance_severity', 'sleep_paralysis_severity', 'sleepiness_severity', 'slow_growth_severity',
    'slow_healing_severity', 'slow_healing_sores_severity', 'slow_heart_rate_severity',
    'slow_heartbeat_severity', 'slow_reading_severity', 'slowness_movement_severity',
    'slurred_speech_severity', 'smooth_bald_patches_severity', 'sneezing_severity', 'snoring_severity',
    'social_communication_difficulty_severity', 'social_problems_severity', 'social_withdrawal_severity',
    'soft_lump_severity', 'sore_throat_severity', 'soreness_severity', 'sores_from_scratching_severity',
    'sound_sensitivity_severity', 'speech_difficulty_severity', 'spelling_problems_severity',
    'spreading_rash_severity', 'staining_severity', 'staring_spells_severity', 'stiff_neck_severity',
    'stiffness_severity', 'stomach_discomfort_severity', 'stooped_posture_severity', 'straining_severity',
    'sudden_chest_pain_severity', 'sudden_collapse_severity', 'sudden_severe_headache_severity',
    'sudden_sleep_attacks_severity', 'sudden_warmth_severity', 'sweating_severity', 'swelling_severity',
    'swelling_anal_area_severity', 'swelling_around_eye_severity', 'swelling_around_tooth_severity',
    'swelling_eyelid_severity', 'swelling_face_severity', 'swelling_inner_ankle_severity',
    'swelling_near_bone_severity', 'swelling_throat_severity', 'swollen_eyelids_severity',
    'swollen_gums_severity', 'swollen_lymph_nodes_severity', 'swollen_salivary_glands_severity',
    'swollen_spleen_severity', 'taste_changes_severity', 'tearing_severity', 'tender_abdomen_severity',
    'tender_gums_severity', 'tender_skin_severity', 'tenderness_severity', 'tenderness_elbow_severity',
    'tenderness_heel_severity', 'tendon_pain_severity', 'testicular_pain_severity', 'thick_mucus_severity',
    'thick_skin_severity', 'thickened_nails_severity', 'thickened_skin_severity', 'thinning_hair_severity',
    'thirst_severity', 'throat_irritation_severity', 'tightness_severity', 'tingling_severity',
    'tingling_hands_feet_severity', 'tingling_scalp_severity', 'tinnitus_severity', 'tired_eyes_severity',
    'toe_bump_severity', 'toe_pain_severity', 'tolerance_severity', 'tooth_pain_severity',
    'toothache_severity', 'tremor_severity', 'tremor_worse_with_movement_severity', 'tunnel_vision_severity',
    'turning_up_volume_severity', 'twisted_foot_severity', 'undescended_testicle_severity',
    'uneven_shoulders_severity', 'uneven_waist_severity', 'unpleasant_taste_severity',
    'unrefreshing_sleep_severity', 'unstable_relationships_severity', 'urgency_severity',
    'urgency_defecate_severity', 'urinary_problems_severity', 'urinary_urgency_severity',
    'urine_leakage_severity', 'vaginal_bleeding_severity', 'vaginal_discharge_severity', 'vertigo_severity',
    'visible_blood_vessels_severity', 'visible_bulge_severity', 'visible_deformity_severity',
    'visible_holes_severity', 'visible_lump_severity', 'visible_muscle_hardening_severity',
    'visible_scalp_veins_severity', 'visible_spine_curve_severity', 'vision_loss_severity',
    'vision_problems_severity', 'voice_loss_severity', 'voice_tremor_severity', 'vomiting_severity',
    'vomiting_after_cough_severity', 'warmth_severity', 'warmth_at_joint_severity', 'warmth_at_site_severity',
    'watery_diarrhea_severity', 'watery_eyes_severity', 'weak_grip_severity', 'weak_pulse_severity',
    'weak_pulse_legs_severity', 'weak_urine_stream_severity', 'weakness_severity', 'weakness_hand_severity',
    'weakness_one_side_severity', 'weight_changes_severity', 'weight_gain_severity',
    'weight_gain_upper_body_severity', 'weight_loss_severity', 'welts_severity', 'wheezing_severity',
    'white_patches_severity', 'white_patches_mouth_severity', 'white_patches_skin_severity',
    'white_round_lesions_severity', 'whiteheads_severity', 'whooping_sound_severity',
    'widespread_pain_severity', 'withdrawal_severity', 'withdrawal_symptoms_severity', 'wrist_pain_severity',
    'writing_changes_severity'
]

# Duration columns
DURATION_COLUMNS = [
    'fever_days', 'cough_days', 'fatigue_days', 'headache_days',
    'diarrhea_days', 'shortness_of_breath_days', 'nausea_days', 'body_ache_days'
]

# One-hot encoded columns (after pd.get_dummies with drop_first=True)
# Gender: Female is dropped (baseline), Male is kept
# Smoking Status: Current is dropped (baseline), Former and Never are kept
CATEGORICAL_ENCODED_COLUMNS = ['gender_Male', 'smoking_status_Former', 'smoking_status_Never']

# All feature columns in the correct order for the model
# Total: 832 columns = age (1) + bmi (1) + symptoms (819) + duration (8) + categorical (3)
ALL_FEATURE_COLUMNS = (
    ['age', 'bmi'] + 
    SYMPTOM_COLUMNS + 
    DURATION_COLUMNS + 
    CATEGORICAL_ENCODED_COLUMNS
)

# ============================================
# COMMON SYMPTOMS FOR USER INPUT
# These are the main symptoms shown in the form
# ============================================

COMMON_SYMPTOMS = [
    ('fever_severity', 'Fever'),
    ('cough_severity', 'Cough'),
    ('fatigue_severity', 'Fatigue'),
    ('headache_severity', 'Headache'),
    ('body_ache_severity', 'Body Ache'),
    ('nausea_severity', 'Nausea'),
    ('diarrhea_severity', 'Diarrhea'),
    ('shortness_of_breath_severity', 'Shortness of Breath'),
    ('chest_pain_severity', 'Chest Pain'),
    ('joint_pain_severity', 'Joint Pain'),
    ('muscle_weakness_severity', 'Muscle Weakness'),
    ('dizziness_severity', 'Dizziness'),
    ('vomiting_severity', 'Vomiting'),
    ('sore_throat_severity', 'Sore Throat'),
    ('runny_nose_severity', 'Runny Nose'),
    ('skin_rash_severity', 'Skin Rash'),
    ('abdominal_pain_severity', 'Abdominal Pain'),
    ('back_pain_severity', 'Back Pain'),
    ('anxiety_severity', 'Anxiety'),
    ('depression_severity', 'Depression'),
]

# ============================================
# HELPER FUNCTIONS
# ============================================

def preprocess_input(form_data):
    """
    Preprocess user input to match the exact training data structure.
    
    Steps:
    1. Create a DataFrame with all required columns initialized to 0
    2. Fill in the values provided by the user
    3. Apply one-hot encoding for categorical variables
    4. Ensure column order matches training data
    
    Args:
        form_data: Dictionary containing user input from the form
        
    Returns:
        pandas DataFrame with exactly 832 columns ready for prediction
    """
    
    # Step 1: Create a template DataFrame with all columns set to 0
    input_dict = {col: [0] for col in ALL_FEATURE_COLUMNS}
    input_df = pd.DataFrame(input_dict)
    
    # Step 2: Fill in basic patient information
    try:
        input_df['age'] = float(form_data.get('age', 30))
        input_df['bmi'] = float(form_data.get('bmi', 22))
    except (ValueError, TypeError):
        input_df['age'] = 30.0
        input_df['bmi'] = 22.0
    
    # Step 3: Handle gender encoding
    # Training used: pd.get_dummies(columns=['gender'], drop_first=True)
    # With drop_first=True, Female is dropped (baseline), Male is kept
    gender = form_data.get('gender', 'Female')
    input_df['gender_Male'] = 1 if gender == 'Male' else 0
    
    # Step 4: Handle smoking status encoding
    # Training used: pd.get_dummies(columns=['smoking_status'], drop_first=True)
    # With drop_first=True, Current is dropped (baseline), Former and Never are kept
    smoking = form_data.get('smoking_status', 'Never')
    input_df['smoking_status_Former'] = 1 if smoking == 'Former' else 0
    input_df['smoking_status_Never'] = 1 if smoking == 'Never' else 0
    
    # Step 5: Fill in symptom severity values
    for symptom_col, _ in COMMON_SYMPTOMS:
        try:
            value = int(form_data.get(symptom_col, 0))
            if symptom_col in input_df.columns:
                input_df[symptom_col] = min(max(value, 0), 10)  # Clamp between 0-10
        except (ValueError, TypeError):
            pass
    
    # Step 6: Fill in duration values
    for duration_col in DURATION_COLUMNS:
        try:
            value = int(form_data.get(duration_col, 0))
            if duration_col in input_df.columns:
                input_df[duration_col] = max(value, 0)
        except (ValueError, TypeError):
            pass
    
    # Step 7: Ensure columns are in the correct order
    input_df = input_df[ALL_FEATURE_COLUMNS]
    
    return input_df


def validate_input(form_data):
    """
    Validate user input and return appropriate error messages.
    
    Args:
        form_data: Dictionary containing user input
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []
    
    # Validate age
    try:
        age = float(form_data.get('age', 0))
        if age < 0 or age > 120:
            errors.append("Age must be between 0 and 120 years.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid age.")
    
    # Validate BMI
    try:
        bmi = float(form_data.get('bmi', 0))
        if bmi < 10 or bmi > 60:
            errors.append("BMI must be between 10 and 60.")
    except (ValueError, TypeError):
        errors.append("Please enter a valid BMI.")
    
    # Validate gender
    if form_data.get('gender') not in ['Male', 'Female']:
        errors.append("Please select a valid gender.")
    
    # Validate smoking status
    if form_data.get('smoking_status') not in ['Never', 'Former', 'Current']:
        errors.append("Please select a valid smoking status.")
    
    if errors:
        return False, " ".join(errors)
    return True, ""


# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    """
    Render the main page with the health assessment form.
    """
    return render_template(
        'index.html',
        symptoms=COMMON_SYMPTOMS,
        duration_columns=DURATION_COLUMNS
    )


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission and make disease prediction.
    
    Steps:
    1. Validate input data
    2. Preprocess input to match training structure
    3. Make prediction using the loaded model with probabilities
    4. Get top 3 predictions with confidence scores
    5. Return result to user
    """
    
    # Check if model is loaded
    if model is None or label_encoder is None:
        return render_template(
            'index.html',
            symptoms=COMMON_SYMPTOMS,
            duration_columns=DURATION_COLUMNS,
            error="Model not loaded. Please check that disease_model.pkl and label_encoder.pkl are in the correct location.",
            show_result=False
        )
    
    # Get form data
    form_data = request.form.to_dict()
    
    # Validate input
    is_valid, error_msg = validate_input(form_data)
    if not is_valid:
        return render_template(
            'index.html',
            symptoms=COMMON_SYMPTOMS,
            duration_columns=DURATION_COLUMNS,
            error=error_msg,
            show_result=False,
            form_data=form_data
        )
    
    try:
        # Preprocess input to match training data structure
        input_df = preprocess_input(form_data)
        
        # Verify we have the correct number of features
        if len(input_df.columns) != 832:
            raise ValueError(f"Expected 832 features, got {len(input_df.columns)}")
        
        # Get prediction probabilities for all classes
        probabilities = model.predict_proba(input_df)[0]
        
        # Get indices of top 3 predictions (sorted by probability descending)
        top_indices = np.argsort(probabilities)[::-1][:3]
        
        # Helper function to get confidence level
        def get_confidence_level(confidence):
            if confidence > 40:
                return 'High Match'
            elif confidence >= 20:
                return 'Moderate Match'
            else:
                return 'Possible Match'
        
        # Build top 3 predictions with confidence scores
        top_predictions = []
        for idx in top_indices:
            disease_name = label_encoder.inverse_transform([idx])[0]
            confidence = probabilities[idx] * 100  # Convert to percentage
            confidence_level = get_confidence_level(confidence)
            top_predictions.append({
                'disease': disease_name,
                'confidence': round(confidence, 1),
                'level': confidence_level
            })
        
        # Primary prediction (highest confidence)
        predicted_disease = top_predictions[0]['disease']
        primary_confidence = top_predictions[0]['confidence']
        primary_level = top_predictions[0]['level']
        
        # Extract selected symptoms for display
        selected_symptoms = []
        for severity_col, symptom_name in COMMON_SYMPTOMS:
            # Check if the symptom was selected (has_* checkbox or severity > 0)
            has_key = 'has_' + severity_col.replace('_severity', '')
            if form_data.get(has_key) or (form_data.get(severity_col) and int(form_data.get(severity_col, 0)) > 0):
                selected_symptoms.append(symptom_name)

        ai_guidance = generate_grok_guidance(top_predictions, selected_symptoms, form_data)
        
        # Return result with all predictions
        return render_template(
            'index.html',
            symptoms=COMMON_SYMPTOMS,
            duration_columns=DURATION_COLUMNS,
            prediction=predicted_disease,
            primary_confidence=primary_confidence,
            primary_level=primary_level,
            top_predictions=top_predictions,
            selected_symptoms=selected_symptoms,
            ai_guidance=ai_guidance,
            show_result=True,
            form_data=form_data
        )
        
    except Exception as e:
        # Handle any prediction errors
        return render_template(
            'index.html',
            symptoms=COMMON_SYMPTOMS,
            duration_columns=DURATION_COLUMNS,
            error=f"Prediction error: {str(e)}",
            show_result=False,
            form_data=form_data
        )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for disease prediction (JSON response).
    Useful for integration with other applications.
    Returns top 3 predictions with confidence scores.
    """
    
    if model is None or label_encoder is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input
        is_valid, error_msg = validate_input(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Preprocess and predict with probabilities
        input_df = preprocess_input(data)
        probabilities = model.predict_proba(input_df)[0]
        
        # Helper function to get confidence level
        def get_confidence_level(confidence):
            if confidence > 40:
                return 'High Match'
            elif confidence >= 20:
                return 'Moderate Match'
            else:
                return 'Possible Match'
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        for idx in top_indices:
            disease_name = label_encoder.inverse_transform([idx])[0]
            confidence = round(probabilities[idx] * 100, 1)
            top_predictions.append({
                'disease': disease_name,
                'confidence': confidence,
                'level': get_confidence_level(confidence)
            })

        selected_symptoms = []
        for severity_col, symptom_name in COMMON_SYMPTOMS:
            has_key = 'has_' + severity_col.replace('_severity', '')
            if data.get(has_key) or (data.get(severity_col) and int(data.get(severity_col, 0)) > 0):
                selected_symptoms.append(symptom_name)

        ai_guidance = generate_grok_guidance(top_predictions, selected_symptoms, data)
        
        return jsonify({
            'success': True,
            'prediction': top_predictions[0]['disease'],
            'confidence': top_predictions[0]['confidence'],
            'confidence_level': top_predictions[0]['level'],
            'top_predictions': top_predictions,
            'guidance': ai_guidance,
            'message': f"Based on your symptoms, you may be showing patterns similar to: {top_predictions[0]['disease']} ({top_predictions[0]['level']})",
            'disclaimer': 'This is not a medical diagnosis. Please consult a healthcare professional.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """
    Health check endpoint for monitoring.
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'encoder_loaded': label_encoder is not None,
        'grok_enabled': bool(GROK_API_KEY)
    })


# ============================================
# RUN APPLICATION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Healthcare Chatbot - Disease Prediction System")
    print("="*60)
    print(f"\n  Model path: {MODEL_PATH}")
    print(f"  Encoder path: {ENCODER_PATH}")
    print(f"  Model loaded: {'Yes' if model else 'No'}")
    print(f"  Encoder loaded: {'Yes' if label_encoder else 'No'}")
    print(f"\n  Starting server at http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    # Run the Flask development server
    # In production, use a proper WSGI server like gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
