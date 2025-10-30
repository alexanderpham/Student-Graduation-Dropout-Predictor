import tensorflow as tf
import numpy as np

label_names = ['android', 'javascript', 'python', 'c#', 'jquery', 'java', 'r', 'mysql', 'c++', 'php']


model = tf.keras.models.load_model("final_model.keras")

def classify(titles: list[str]) -> list[str]:
    input_tensor = tf.convert_to_tensor(titles, dtype=tf.string)
    predictions = model.predict(input_tensor)
    predicted_indices = np.argmax(predictions, axis=1)
    return [label_names[i] for i in predicted_indices]


if __name__ == "__main__":
    sample_titles = [
        "How to center a div using CSS",
        "Fixing a NullPointerException in Java",
        "Connect Python to a MySQL database",
        "Difference between JavaScript and jQuery",
        "Recursive function in C++"
    ]

    results = classify(sample_titles)

    for i in range(len(sample_titles)):
        print(f"{sample_titles[i]} → {results[i]}")
