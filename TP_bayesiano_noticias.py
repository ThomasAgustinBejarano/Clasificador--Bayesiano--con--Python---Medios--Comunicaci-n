from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------
# 1. Definición del Dataset
# ---------------------------
data = {
    'noticias': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"],
    'label': [
        'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake'
    ]
}

# Creación del DataFrame de trabajo
df = pd.DataFrame(data)

# ------------------------------------
# 2. Preprocesamiento y Vectorización 
# ------------------------------------
X = df['noticias']
y = df['label']

# Conversión del texto a matriz numérica de frecuencias
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


# -----------------------------------------------
# 3. División del Dataset (Entrenamiento y Prueba 
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# ---------------------------
# 4. Entrenamiento del Modelo
# ---------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluación del Rendimiento
# -----------------------------
y_pred = model.predict(X_test)

print("=" * 50)
print("Reporte de clasificación:")
print(" ")
print(classification_report(y_test, y_pred))

print("=" * 50)
print("Matriz de confusión:")
print(" ")
print(confusion_matrix(y_test, y_pred))


# -----------------------------------
# 6. Clasificación de Nuevos Mensajes
# -----------------------------------
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

# Vectorización de los nuevos mensajes y predicción
nuevas_vectorizadas = vectorizer.transform(nuevas_noticias)
predicciones = model.predict(nuevas_vectorizadas)

print("=" * 50)
print("Clasificación de Nuevas Noticias:")
print(" ")
for noticia, prediccion in zip(nuevas_noticias, predicciones):
    print(f'Noticia: "{noticia}" → Clasificación: {prediccion}')

