"""
Gamification Routes for FitWrk Backend
Handles challenge generation, exercise verification, and lift verification
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import tempfile
import requests
import os
import google.generativeai as genai

# Gamification imports
from models.movenet_detector import MoveNetDetector
from utils.video_processor import VideoProcessor
from analyzers.universal_exercise_analyzer import UniversalExerciseAnalyzer

# Configure Gemini
genai.configure(api_key=os.environ.get('GEMINI_API_KEY', ''))

# Create router
router = APIRouter(prefix="/api/gamification", tags=["gamification"])

# Lazy-load MoveNet (only initialize on first use for faster startup)
detector = None
video_processor = None

def get_video_processor():
    """Lazy-load video processor to speed up Cloud Run startup"""
    global detector, video_processor
    if video_processor is None:
        print("="*50)
        print("Initializing MoveNet detector (first use)...")
        print("="*50)
        detector = MoveNetDetector(model_variant='lightning')
        video_processor = VideoProcessor(detector=detector)
        print("="*50)
        print("MoveNet ready!")
        print("="*50)
    return video_processor

# Request/Response Models
class VerifyExerciseRequest(BaseModel):
    videoUrl: str
    exerciseType: str
    targetMetric: str

class GenerateChallengeRequest(BaseModel):
    muscleGroup: str
    fitnessLevel: str
    previousChallenges: list = []

class VerifyLiftRequest(BaseModel):
    videoUrl: str
    exerciseName: str

class GenerateScheduleRequest(BaseModel):
    fitnessLevel: str
    goals: list[str]
    daysPerWeek: int
    preferences: dict = {}

# Helper Functions
def download_video(url: str) -> str:
    """Download video from URL to temporary file"""
    response = requests.get(url, stream=True, timeout=60)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            temp_file.write(chunk)

    temp_file.close()
    return temp_file.name

def generate_feedback(exercise_type: str, reps: int, target: int, form_score: float, detailed: dict) -> str:
    """Generate feedback text"""
    feedback_parts = []

    # Rep count feedback
    if reps >= target:
        feedback_parts.append(f"Great! Completed {reps}/{target} reps.")
    else:
        feedback_parts.append(f"Completed {reps}/{target} reps. Keep trying!")

    # Form feedback
    if form_score >= 80:
        feedback_parts.append("Excellent form!")
    elif form_score >= 70:
        feedback_parts.append("Good form with minor improvements needed.")
    else:
        feedback_parts.append("Form needs improvement.")

    # Specific feedback
    if detailed.get('body_alignment', 100) < 70:
        feedback_parts.append("Keep your body in a straight line.")
    if detailed.get('range_of_motion', 100) < 70:
        feedback_parts.append("Go lower for full range of motion.")
    if detailed.get('stability', 100) < 70:
        feedback_parts.append("Reduce side-to-side movement.")

    return " ".join(feedback_parts)

# Routes

@router.post("/verify-exercise")
async def verify_exercise(request: VerifyExerciseRequest):
    """
    Verify exercise video using MoveNet

    Returns verification result with form score and rep count
    """
    try:
        print("\n" + "="*50)
        print("NEW VERIFICATION REQUEST")
        print("="*50)

        exercise_type = request.exerciseType.lower()
        target_metric = request.targetMetric

        # Extract target reps
        target_reps = int(''.join(filter(str.isdigit, target_metric)))
        print(f"Exercise: {exercise_type}")
        print(f"Target: {target_reps} reps")

        # Download video
        print("Downloading video...")
        temp_video = download_video(request.videoUrl)
        print(f"Downloaded to: {temp_video}")

        # Get video processor (lazy-loaded)
        vp = get_video_processor()

        # Process video with MoveNet
        print("Processing with MoveNet...")
        all_keypoints, fps, frame_count = vp.process_video(
            temp_video,
            frame_skip=2  # 30fps -> 15fps
        )

        # Use universal analyzer (works for ANY exercise!)
        analyzer = UniversalExerciseAnalyzer(detector.KEYPOINT_DICT)

        print("Counting reps...")
        reps_detected, angles, rep_frames = analyzer.count_reps(
            all_keypoints,
            exercise_type=None,  # Auto-detect
            exercise_hint=exercise_type
        )

        print("Analyzing form...")
        form_score, detailed_scores = analyzer.analyze_form(
            all_keypoints,
            exercise_type=None,  # Auto-detect
            exercise_hint=exercise_type
        )

        # Determine if verified
        verified = (
            form_score >= 70 and
            reps_detected >= target_reps * 0.9  # Allow 10% margin
        )

        # Generate feedback
        feedback = generate_feedback(
            exercise_type,
            reps_detected,
            target_reps,
            form_score,
            detailed_scores
        )

        # Cleanup
        os.remove(temp_video)

        print("="*50)
        print("RESULT:")
        print(f"  Reps: {reps_detected}/{target_reps}")
        print(f"  Form Score: {form_score:.1f}%")
        print(f"  Verified: {verified}")
        print("="*50)

        return {
            "verified": verified,
            "formScore": round(form_score, 1),
            "repsDetected": reps_detected,
            "targetReps": target_reps,
            "detailedScores": detailed_scores,
            "feedback": feedback
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print("="*50)
        print("ERROR:")
        print(str(e))
        print("="*50)
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/generate-challenge")
async def generate_challenge(request: GenerateChallengeRequest):
    """
    Generate AI workout challenge using Gemini

    Returns challenge with title, description, target metric, and difficulty
    """
    try:
        muscle_group = request.muscleGroup
        fitness_level = request.fitnessLevel
        previous_challenges = request.previousChallenges

        print(f"Generating challenge: {muscle_group} for {fitness_level}")

        # Create prompt for Gemini
        previous_titles = ', '.join([c.get('title', '') for c in previous_challenges]) if previous_challenges else "None"

        prompt = f"""You are a professional fitness coach. Generate a unique, exciting workout challenge.

Muscle Group: {muscle_group}
Fitness Level: {fitness_level}
Previous Challenges: {previous_titles}

Create a NEW challenge that's different from previous ones. Return ONLY valid JSON (no markdown, no code blocks):

{{
  "title": "Epic Challenge Name",
  "description": "Detailed description of what to do (2-3 sentences)",
  "targetMetric": "specific measurable goal (e.g., '25 pushups', '5k run', '100kg bench')",
  "difficulty": "easy, medium, or hard"
}}

Guidelines:
- Make title exciting and motivating
- Description should be clear and specific
- Target metric must be measurable for verification
- Difficulty should match {fitness_level} level
- Be creative and vary from previous challenges
- For {muscle_group}, choose appropriate exercises

Respond with ONLY the JSON object, no other text."""

        # Call Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        challenge_text = response.text.strip()

        # Remove markdown code blocks if present
        if challenge_text.startswith('```'):
            challenge_text = challenge_text.split('```')[1]
            if challenge_text.startswith('json'):
                challenge_text = challenge_text[4:]
            challenge_text = challenge_text.strip()

        # Parse JSON
        import json
        challenge_data = json.loads(challenge_text)

        print(f"Generated: {challenge_data.get('title', 'Unknown')}")

        return challenge_data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Raw response: {challenge_text}")
        # Fallback to hardcoded challenge
        return {
            'title': f'{muscle_group.title()} Challenge',
            'description': f'Complete a challenging {muscle_group} workout',
            'targetMetric': '20 reps',
            'difficulty': 'medium'
        }
    except Exception as e:
        print(f"Error generating challenge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify-lift")
async def verify_lift(request: VerifyLiftRequest):
    """
    Verify lift record (similar to verify-exercise but stricter)

    Returns verification result with form score
    """
    # For now, redirect to verify-exercise logic
    # TODO: Implement stricter verification for lifts
    verify_request = VerifyExerciseRequest(
        videoUrl=request.videoUrl,
        exerciseType=request.exerciseName,
        targetMetric="1 rep"  # Lifts typically 1-5 reps
    )

    return await verify_exercise(verify_request)

@router.post("/generate-schedule")
async def generate_schedule(request: GenerateScheduleRequest):
    """
    Generate AI-powered workout schedule using Gemini

    Returns a personalized workout schedule based on fitness level, goals, and preferences
    """
    try:
        fitness_level = request.fitnessLevel
        goals = request.goals
        days_per_week = request.daysPerWeek
        preferences = request.preferences

        print(f"Generating schedule: {fitness_level}, {days_per_week} days/week, goals: {goals}")

        # Create detailed prompt for Gemini
        prompt = f"""You are a professional fitness coach. Create a detailed {days_per_week}-day workout schedule.

Fitness Level: {fitness_level}
Goals: {', '.join(goals)}
Days per week: {days_per_week}
Preferences: {preferences}

Return ONLY a valid JSON object with this exact structure (no markdown, no code blocks):
{{
  "schedule": [
    {{
      "day": "Monday",
      "focus": "Chest & Triceps",
      "exercises": [
        {{"name": "Bench Press", "sets": 3, "reps": "8-10"}},
        {{"name": "Incline Dumbbell Press", "sets": 3, "reps": "10-12"}},
        {{"name": "Cable Flyes", "sets": 3, "reps": "12-15"}},
        {{"name": "Tricep Dips", "sets": 3, "reps": "10-12"}},
        {{"name": "Tricep Pushdowns", "sets": 3, "reps": "12-15"}}
      ],
      "restBetweenSets": "60-90 seconds",
      "notes": "Focus on proper form and controlled movements"
    }}
  ],
  "weeklyOverview": "This schedule targets all major muscle groups...",
  "nutritionTips": "Consume 0.8-1g protein per lb of bodyweight...",
  "restDays": ["Wednesday", "Sunday"]
}}

Important:
- Respond ONLY with valid JSON
- No markdown formatting (no ```json or ```)
- Include {days_per_week} workout days
- Vary exercises for muscle groups
- Include proper rest days
- Make it suitable for {fitness_level} level"""

        # Call Gemini
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        schedule_text = response.text.strip()

        # Remove markdown code blocks if present
        if schedule_text.startswith('```'):
            schedule_text = schedule_text.split('```')[1]
            if schedule_text.startswith('json'):
                schedule_text = schedule_text[4:]
            schedule_text = schedule_text.strip()

        # Parse JSON
        import json
        schedule_data = json.loads(schedule_text)

        print("Schedule generated successfully!")
        return schedule_data

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Raw response: {schedule_text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse AI response: {str(e)}"
        )
    except Exception as e:
        print(f"Error generating schedule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/health")
async def health():
    """Health check for gamification system"""
    return {
        "status": "healthy",
        "model": "MoveNet Lightning",
        "service": "FitWrk Gamification"
    }
