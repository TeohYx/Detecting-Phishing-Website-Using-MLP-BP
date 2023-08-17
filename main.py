import tkinter as tk
from tkinter import messagebox
import numpy as np
from sklearn.neural_network import MLPClassifier
import joblib

def main():
    # Load the trained model
    model = joblib.load('finalized_model.pkl')

    # Define the feature names
    feature_names = ['SFH', 'popUpWidnow', 'SSLfinal_State', 'Request_URL', 'URL_of_Anchor', 'web_traffic', 'URL_Length', 'age_of_domain', 'having_IP_Address']

    # Define the mapping of labels to values
    label_mapping = {-1: 'Phishy', 0: 'Suspicious', 1: 'Legitimate'}

    # Function to handle the button click event for manual input
    def manual_input():
        # Create a new window
        input_window = tk.Toplevel(root)
        input_window.title('Input Menu')

        # Create labels and radio buttons for each feature
        entries = []
        for i, feature in enumerate(feature_names):
            label = tk.Label(input_window, text=feature)
            label.grid(row=i, column=0, padx=10, pady=5)

            var = tk.IntVar()  # Variable to store the selected value
            var.set(0)  # Set the default value to 0

            # Create radio buttons for each value (-1, 0, 1)
            for value, text in label_mapping.items():
                radio_button = tk.Radiobutton(input_window, text=text, variable=var, value=value)
                radio_button.grid(row=i, column=value + 4, padx=5, pady=5)

            entries.append(var)

        # Function to handle the predict button click event
        def predict():
            # Get the user inputs
            input_data = [var.get() for var in entries]

            # Create the input dictionary
            input_dict = dict(zip(feature_names, input_data))

            # Convert the input dictionary to a 2D array
            input_array = np.array([list(input_dict.values())])

            # Perform the prediction using the model
            predicted_class = model.predict(input_array)[0]

            # Get the label for the predicted class
            predicted_label = label_mapping[predicted_class]

            # Display the predicted class in a message box
            messagebox.showinfo('Prediction Result', f'Predicted Class: {predicted_label}')

            # Close the input window
            input_window.destroy()

        # Create the predict button
        predict_button = tk.Button(input_window, text='Predict', command=predict)
        predict_button.grid(row=len(feature_names), column=1, columnspan=len(label_mapping) + 1, padx=10, pady=5)

    # Create the main window
    root = tk.Tk()
    root.title('Main Menu')
    root.geometry('300x80')

    # Create a button for entering data
    enter_data_button = tk.Button(root, text='Enter Data', command=manual_input)
    enter_data_button.pack(pady=20)

    # Run the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
