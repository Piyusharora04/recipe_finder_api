import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

df['recipe_id'] = df.index+1

# Preprocess and vectorize ingredients (using TF-IDF for better weighting)
vectorizer = TfidfVectorizer(stop_words='english')
recipe_matrix = vectorizer.fit_transform(df['Cleaned-Ingredients'])



def create_empty_response_json():
    """Creates a JSON response for cases with no matching recipes"""
    empty_data = {
       'recipe_id': [''], 
       'TranslatedRecipeName': [''],
       'Cleaned-Ingredients': [''],
       'TotalTimeInMins': [0], 
       'Cuisine': [''], 
       'TranslatedInstructions': [''],
       'URL': [''],
       'image-url': [''],
       'Ingredient-count': [0]
    }
    return pd.DataFrame(empty_data).to_json(orient='records')



def find_similar_recipes(input_ingredients):
    """Finds the top 20 similar recipes based on input ingredients"""
    input_str = ' '.join(input_ingredients)  # Combine ingredients into a string
    input_vector = vectorizer.transform([input_str])
    similarities = cosine_similarity(input_vector, recipe_matrix)[0]

    top_indices = similarities.argsort()[-50:][::-1]
    
    
    # Filter for recipes with matching ingredients
    matching_recipes = []
    for index in top_indices:
        recipe_ingredients = df.iloc[index]['Cleaned-Ingredients']
        if any(ingredient in recipe_ingredients for ingredient in input_ingredients):
            matching_recipes.append(df.iloc[index])

    # Create recommendations DataFrame or empty response
    if matching_recipes:
        recommendations = pd.DataFrame(matching_recipes)[['recipe_id', 'TranslatedRecipeName' ,'Cleaned-Ingredients','TotalTimeInMins','Cuisine','TranslatedInstructions','URL', 'image-url','Ingredient-count']]
        return recommendations.to_json(orient='records')
    else:
        return create_empty_response_json()  # From your previous code
    
    
from flask import Flask, request

app = Flask(__name__)

@app.route('/finder', methods=['POST'])
def finder():
    input_ingredients = request.get_json()

    
    with open('result.json', 'w') as fp:
        json.dump(input_ingredients, fp)
    
    
    with open('result.json', 'r') as f:
        ingredients_data = json.load(f)
        input_ingr = ' '.join(ingredients_data['ingredients'])
        
        
    recommendations = find_similar_recipes(input_ingr)
    
#     with open("recommendations.json", "w") as f:
#         f.write(recommendations)
    
    return recommendations , 201

@app.route('/')
def welcome():
    return "Welcome to this recipe finder..."