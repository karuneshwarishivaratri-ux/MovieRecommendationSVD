
MOVIE RECOMMENDATION SYSTEM USING SVD
=====================================

PROJECT DESCRIPTION
This project recommends movies to users using
Singular Value Decomposition (SVD) collaborative filtering.

TECHNOLOGIES USED
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-Surprise
- MovieLens 100K Dataset

PROJECT FILES
- app.py: Streamlit dashboard
- model.py: Model training and evaluation script
- ml-100k/: MovieLens dataset
- requirements.txt: Required Python libraries

HOW TO RUN THE PROJECT
1. Install Python on your computer.
2. Open Command Prompt inside the project folder.
3. Install the dependencies:
   python -m pip install -r requirements.txt
4. Start the dashboard:
   python -m streamlit run app.py
5. Open the local URL shown in the terminal,
   usually http://localhost:8501.

FEATURES
- User-based movie recommendations
- Top 5 and Top 10 recommendations
- Predicted movie ratings
- Exclusion of movies already rated by the user
- RMSE and MAE model evaluation
- Comparison with a mean-rating baseline

DATASET
MovieLens 100K dataset.

NOTE
The first installation of scikit-surprise may require
a compatible Python version and build tools.
