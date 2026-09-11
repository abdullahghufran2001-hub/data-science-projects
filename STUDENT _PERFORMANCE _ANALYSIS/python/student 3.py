import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# AI-POWERED STUDENT PERFORMANCE PREDICTION SYSTEM
# ============================================================

file_path = r"C:\Users\user\Downloads\sample_students.csv"


try:

    # ========================================================
    # LOAD DATASET
    # ========================================================

    df = pd.read_csv(file_path)

    print("\n" + "=" * 70)
    print("       AI-POWERED STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("=" * 70)

    print("\nDataset loaded successfully!")

    # ========================================================
    # DISPLAY DATASET
    # ========================================================

    print("\nSTUDENT DATA")
    print("-" * 70)
    print(df.to_string(index=False))

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        "Attendance",
        "StudyHours",
        "PreviousMarks",
        "AssignmentScore",
        "TestScore",
        "Result"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        print("\nERROR: Missing columns:")
        print(missing_columns)

    else:

        # ====================================================
        # FEATURES
        # ====================================================

        features = [
            "Attendance",
            "StudyHours",
            "PreviousMarks",
            "AssignmentScore",
            "TestScore"
        ]

        X = df[features]
        y = df["Result"]

        # ====================================================
        # TRAIN TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ====================================================
        # RANDOM FOREST MODEL
        # ====================================================

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        print("\n" + "=" * 70)
        print("                    MODEL PERFORMANCE")
        print("=" * 70)

        print("\nAlgorithm      : Random Forest Classifier")
        print(
            "Model Accuracy :",
            round(accuracy * 100, 2),
            "%"
        )

        # ====================================================
        # STUDENT INPUT
        # ====================================================

        print("\n" + "=" * 70)
        print("                  ENTER STUDENT DETAILS")
        print("=" * 70)

        attendance = float(
            input("\nAttendance (%): ")
        )

        study_hours = float(
            input("Study Hours: ")
        )

        previous_marks = float(
            input("Previous Marks: ")
        )

        assignment_score = float(
            input("Assignment Score: ")
        )

        test_score = float(
            input("Test Score: ")
        )

        # ====================================================
        # INPUT VALIDATION
        # ====================================================

        if not 0 <= attendance <= 100:
            raise ValueError(
                "Attendance must be between 0 and 100."
            )

        if not 0 <= study_hours <= 20:
            raise ValueError(
                "Study Hours must be between 0 and 20."
            )

        if not 0 <= previous_marks <= 100:
            raise ValueError(
                "Previous Marks must be between 0 and 100."
            )

        if not 0 <= assignment_score <= 100:
            raise ValueError(
                "Assignment Score must be between 0 and 100."
            )

        if not 0 <= test_score <= 100:
            raise ValueError(
                "Test Score must be between 0 and 100."
            )

        # ====================================================
        # CREATE NEW STUDENT DATA
        # ====================================================

        new_student = pd.DataFrame(
            [[
                attendance,
                study_hours,
                previous_marks,
                assignment_score,
                test_score
            ]],
            columns=features
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            new_student
        )[0]

        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        probabilities = model.predict_proba(
            new_student
        )[0]

        classes = model.classes_

        confidence = max(probabilities) * 100

        # ====================================================
        # FINAL RESULT
        # ====================================================

        print("\n" + "=" * 70)
        print("                    PREDICTION RESULT")
        print("=" * 70)

        print("\nAttendance       :", attendance, "%")
        print("Study Hours      :", study_hours)
        print("Previous Marks   :", previous_marks)
        print("Assignment Score :", assignment_score)
        print("Test Score       :", test_score)

        print("\nPredicted Result :", prediction)
        print(
            "Confidence       :",
            round(confidence, 2),
            "%"
        )

        if str(prediction).lower() == "pass":

            print(
                "\nSTUDENT IS PREDICTED TO PASS."
            )

        else:

            print(
                "\nSTUDENT IS PREDICTED TO FAIL."
            )

        # ====================================================
        # CHART 1 - MODEL ACCURACY
        # ====================================================

        plt.figure(
            figsize=(10, 7)
        )

        accuracy_value = accuracy * 100

        bars = plt.bar(
            ["Random Forest"],
            [accuracy_value],
            width=0.45
        )

        plt.ylim(
            0,
            110
        )

        plt.title(
            "MODEL ACCURACY",
            fontsize=20,
            fontweight="bold",
            pad=20
        )

        plt.ylabel(
            "Accuracy (%)",
            fontsize=13
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.25
        )

        for bar in bars:

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                accuracy_value + 3,
                f"{accuracy_value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold"
            )

        plt.subplots_adjust(
            top=0.88,
            bottom=0.15,
            left=0.12,
            right=0.95
        )

        plt.show()


        # ====================================================
        # CHART 2 - FEATURE IMPORTANCE
        # ====================================================

        importance = model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=True
        )

        # Short names for cleaner chart
        display_features = [
            "Attendance",
            "Study Hours",
            "Previous Marks",
            "Assignment",
            "Test Score"
        ]

        importance_df["Display"] = display_features

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=True
        )

        plt.figure(
            figsize=(11, 7)
        )

        bars = plt.barh(
            importance_df["Display"],
            importance_df["Importance"],
            height=0.55
        )

        plt.title(
            "FEATURE IMPORTANCE",
            fontsize=20,
            fontweight="bold",
            pad=20
        )

        plt.xlabel(
            "Importance Score",
            fontsize=13
        )

        max_importance = max(
            importance_df["Importance"]
        )

        plt.xlim(
            0,
            max_importance + 0.10
        )

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.25
        )

        for bar in bars:

            value = bar.get_width()

            plt.text(
                value + 0.015,
                bar.get_y()
                + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=11,
                fontweight="bold"
            )

        plt.subplots_adjust(
            left=0.25,
            right=0.95,
            top=0.87,
            bottom=0.13
        )

        plt.show()


        # ====================================================
        # CHART 3 - STUDENT PERFORMANCE
        # ====================================================

        student_labels = [
            "Attendance",
            "Previous\nMarks",
            "Assignment",
            "Test\nScore"
        ]

        student_values = [
            attendance,
            previous_marks,
            assignment_score,
            test_score
        ]

        plt.figure(
            figsize=(11, 7)
        )

        bars = plt.bar(
            student_labels,
            student_values,
            width=0.55
        )

        plt.title(
            "STUDENT PERFORMANCE ANALYSIS",
            fontsize=20,
            fontweight="bold",
            pad=20
        )

        plt.ylabel(
            "Score (%)",
            fontsize=13
        )

        plt.ylim(
            0,
            110
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.25
        )

        for bar, value in zip(
            bars,
            student_values
        ):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 3,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold"
            )

        plt.subplots_adjust(
            top=0.87,
            bottom=0.18,
            left=0.10,
            right=0.95
        )

        plt.show()


        # ====================================================
        # CHART 4 - PREDICTION PROBABILITY
        # ====================================================

        probability_values = (
            probabilities * 100
        )

        # Clean class labels
        display_classes = [
            str(c).upper()
            for c in classes
        ]

        plt.figure(
            figsize=(10, 7)
        )

        bars = plt.bar(
            display_classes,
            probability_values,
            width=0.50
        )

        plt.title(
            "AI PREDICTION PROBABILITY",
            fontsize=20,
            fontweight="bold",
            pad=20
        )

        plt.ylabel(
            "Probability (%)",
            fontsize=13
        )

        plt.ylim(
            0,
            110
        )

        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.25
        )

        for bar, value in zip(
            bars,
            probability_values
        ):

            plt.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 3,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold"
            )

        plt.subplots_adjust(
            top=0.87,
            bottom=0.15,
            left=0.12,
            right=0.95
        )

        plt.show()


        # ====================================================
        # PROGRAM COMPLETED
        # ====================================================

        print("\n" + "=" * 70)
        print("                    PROGRAM COMPLETED")
        print("=" * 70)


except FileNotFoundError:

    print("\nERROR: CSV FILE NOT FOUND!")

    print("\nPlease check this location:")
    print(file_path)


except ValueError as e:

    print("\nINPUT ERROR:")
    print(e)


except Exception as e:

    print("\nERROR:")
    print(e)


input("\nPress Enter to exit...")
