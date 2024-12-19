import json
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import optuna
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Objective function for Optuna optimization
def objective(trial):
    # Hyperparameters to optimize
    n_estimators = trial.suggest_int("n_estimators", 50, 200)  # Number of trees
    max_depth = trial.suggest_int("max_depth", 3, 20)  # Maximum depth of the tree
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)  # Min samples required to split a node
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)  # Min samples required at leaf node
    
    # Load dataset
    data = pd.read_csv("Training.csv")
    data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors="ignore")
    X = data.drop("prognosis", axis=1)
    y = data["prognosis"]
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Initialize RandomForestClassifier with hyperparameters from the trial
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy


# Train diagnosis model with hyperparameter optimization using Optuna
def train_diagnosis_model(train_data_path="Training.csv", model_path="diagnosis_model.pkl", encoder_path="label_encoder.pkl"):
    # Create an Optuna study to optimize the model
    study = optuna.create_study(direction="maximize")  # Maximize accuracy
    study.optimize(objective, n_trials=10)  # Run 10 trials

    # Get the best hyperparameters from the study
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")
    
    # Train the model with the best hyperparameters
    data = pd.read_csv(train_data_path)
    data = data.drop(columns=[col for col in data.columns if "Unnamed" in col], errors="ignore")
    X = data.drop("prognosis", axis=1)
    y = data["prognosis"]
    
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the model with the best hyperparameters
    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save the trained model and encoder
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"Model trained with accuracy: {accuracy_score(y_test, y_pred)}")
# Define the LLM for agents
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Symptom Extraction Agent
class SymptomExtractionAgent:
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="""You are an expert medical assistant and a Named Entity Recognition (NER) model. Your task is to understand the user's query and infer symptoms they might be experiencing, even if the symptoms are not mentioned explicitly. The symptoms to look for are listed below. For each symptom, indicate whether it is present (1) or absent (0) based on your understanding of the user's query.

Provide the results as a JSON object with symptom names as keys and 1 or 0 as values.

Symptoms to look for:
itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity,
ulcers_on_tongue, muscle_wasting, vomiting, burning_micturition, spotting_urination, fatigue, weight_gain, anxiety,
cold_hands_and_feets, mood_swings, weight_loss, restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough,
high_fever, sunken_eyes, breathlessness, sweating, dehydration, indigestion, headache, yellowish_skin, dark_urine, nausea,
loss_of_appetite, pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea, mild_fever, yellow_urine,
yellowing_of_eyes, acute_liver_failure, fluid_overload, swelling_of_stomach, swelled_lymph_nodes, malaise, blurred_and_distorted_vision,
phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose, congestion, chest_pain, weakness_in_limbs, fast_heart_rate,
pain_during_bowel_movements, pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness, cramps, bruising, obesity,
swollen_legs, swollen_blood_vessels, puffy_face_and_eyes, enlarged_thyroid, brittle_nails, swollen_extremeties, excessive_hunger,
extra_marital_contacts, drying_and_tingling_lips, slurred_speech, knee_pain, hip_joint_pain, muscle_weakness, stiff_neck, swelling_joints,
movement_stiffness, spinning_movements, loss_of_balance, unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort,
foul_smell_of_urine, continuous_feel_of_urine, passage_of_gases, internal_itching, toxic_look_(typhos), depression, irritability, muscle_pain,
altered_sensorium, red_spots_over_body, belly_pain, abnormal_menstruation, dischromic_patches, watering_from_eyes, increased_appetite,
polyuria, family_history, mucoid_sputum, rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion,
receiving_unsterile_injections, coma, stomach_bleeding, distention_of_abdomen, history_of_alcohol_consumption, fluid_overload.1,
blood_in_sputum, prominent_veins_on_calf, palpitations, painful_walking, pus_filled_pimples, blackheads, scurring, skin_peeling,
silver_like_dusting, small_dents_in_nails, inflammatory_nails, blister, red_sore_around_nose, yellow_crust_ooze."""
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def run(self, query: str) -> dict:
        result = self.chain.run(query=query,)
        print(f"SymptomExtractionAgent result: {result[8:-5]}") 
        return json.loads(result[8:-5])


# Diagnosis Agent
class DiagnosisAgent:
    def __init__(self, model_path="diagnosis_model.pkl",encoder_path = "label_encoder.pkl"):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = self.load_model()
        self.label_encoder = self.load_encoder()

    def load_model(self):
        try:
            with open(self.model_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Model file not found at {self.model_path}. Train the model first.")
        
    def load_encoder(self):
        try:
            with open(self.encoder_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Label encoder file not found at {self.encoder_path}. Train the model first.")

    def run(self, symptoms: dict) -> str:
        input_data = pd.DataFrame([symptoms])
        missing_cols = set(self.model.feature_names_in_) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0
        input_data = input_data[self.model.feature_names_in_]
        prediction = self.model.predict(input_data)[0]
        
        diagnosis_label = self.label_encoder.inverse_transform([prediction])[0]
        print('Prediction',prediction, 'label', diagnosis_label)        
        return diagnosis_label


# Recommendation Agent
class RecommendationAgent:
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["diagnosis"],
            template="""You are an expert medical assistant. Based on the diagnosis {diagnosis}, provide the necessary recommendations for the patient. 
            Ensure that the recommendations are clear, actionable, and tailored to the specific diagnosis. Provide the results as a JSON object with 
            recommendation categories as keys and the corresponding advice as values."""
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def run(self, diagnosis: str) -> dict:
        result = self.chain.run(diagnosis=diagnosis)
        return json.loads(result[8:-5])


# Health System Graph
# ...existing code...

class HealthSystemGraph:
    def __init__(self):
        self.symptom_extraction_agent = SymptomExtractionAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(self, query: str) -> dict:
        # Extract symptoms
        symptoms = self.symptom_extraction_agent.run(query)
        print(f"Extracted symptoms: {symptoms}")

        # Diagnose based on symptoms
        diagnosis = self.diagnosis_agent.run(symptoms)
        print(f"Diagnosis: {diagnosis}")

        # Provide recommendations
        recommendations = self.recommendation_agent.run(diagnosis)
        print(f"Recommendations: {recommendations}")

        return {"symptoms": symptoms, "diagnosis": diagnosis, "recommendations": recommendations}




# Main Function
def main():
    # Optional: Train the model if not already trained
    #train_diagnosis_model()

    # Create the Health System
    health_system = HealthSystemGraph()

    # Run the system with a sample query
    query = "I have been feeling very tired and experiencing a rash and some sneezing."
    result = health_system.run(query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


