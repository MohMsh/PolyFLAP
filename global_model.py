from copy import deepcopy
import numpy as np
from sklearn import metrics


class global_model:

    def __init__(self, models_parameters, global_model_, x_test, y_test, current_round, print_model_summary,
                 print_model_performance):
        self.models_parameters = models_parameters
        self.x_test = x_test
        self.y_test = y_test
        self.global_model_ = global_model_
        self.current_round = current_round
        self.print_model_summary = print_model_summary
        self.print_model_performance = print_model_performance

    def print_round(self):
        print("\n\t==========================================\n"
              "\tGlobal model aggregation initiated (Round ", self.current_round, ")",
              "\n\t==========================================\n")

    def aggregate(self):

        # print some info related to global model
        self.print_round()
        model_type = type(self.global_model_).__name__
        if model_type == "SVC":
            self.global_model_, aggregated_parameters = self.svm_aggregate()
            predicted = self.svm_custom_predict(self.global_model_, self.x_test)
            self.evaluate(predicted) if self.print_model_performance else None
            self.svm_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "LogisticRegression":
            self.global_model_, aggregated_parameters = self.lr_aggregate()
            predicted = self.global_model_.predict(self.x_test)
            self.evaluate(predicted) if self.print_model_performance else None
            self.lr_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "GaussianNB":
            self.global_model_, aggregated_parameters = self.gnb_aggregate()
            predicted = self.global_model_.predict(self.x_test)
            self.evaluate(predicted) if self.print_model_performance else None
            self.gnb_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "SGDClassifier":
            self.global_model_, aggregated_parameters = self.sgd_aggregate()
            predicted = self.global_model_.predict(self.x_test)
            self.evaluate(predicted) if self.print_model_performance else None
            self.sgd_summary(self.global_model_) if self.print_model_summary else None
        elif model_type == "MLPClassifier":
            self.global_model_, aggregated_parameters = self.mlp_aggregate()
            predicted = self.global_model_.predict(self.x_test)
            self.evaluate(predicted) if self.print_model_performance else None
            self.mlp_summary(self.global_model_) if self.print_model_summary else None
        else:
            print("The provided model is not supported yet with this framework")
            return

        return self.global_model_, aggregated_parameters

    def evaluate(self, predicted):
        print("\t<===> Model Performance Metrics: \n\t=============================")
        # evaluate model and print performance metrics
        print("\t\t==> Accuracy: ", round(metrics.accuracy_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Precision: ", round(metrics.precision_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Recall: ", round(metrics.recall_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> F1 Score: ", round(metrics.f1_score(self.y_test, predicted) * 100, 2), "%")
        print("\t\t==> Specificity: ", round(metrics.recall_score
                                             (self.y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t\t==> Negative Predictive Value (NPV): ",
              round(metrics.precision_score(self.y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t=============================")

    def svm_aggregate(self):
        # Initialize the aggregated support vectors, coefficients, and intercept
        aggregated_support_vectors = []
        aggregated_coefficients = [[] for _ in range(self.models_parameters[0]["coefs"].shape[0])]
        aggregated_intercept = 0

        # Combine support vectors, coefficients, and intercepts from each client's model
        for model in self.models_parameters:
            aggregated_support_vectors.extend(model["support_vector"])
            for i in range(model["coefs"].shape[0]):
                aggregated_coefficients[i].extend(model["coefs"][i])
            aggregated_intercept += model["intercept"]

        # Average the intercept
        aggregated_intercept /= len(self.models_parameters)

        # Create a new SVM model with the same kernel, C, and other parameters as the original models
        # this can be done by using the initial model used which is self.model
        aggregated_model = deepcopy(self.global_model_)
        # Set the aggregated support vectors, coefficients, and intercept for the new model
        aggregated_model.support_vectors_ = np.vstack(aggregated_support_vectors)
        aggregated_model.dual_coef_ = np.vstack([np.array(row) for row in aggregated_coefficients])
        aggregated_model.intercept_ = aggregated_intercept

        # Fit the aggregated model with dummy data (to bypass the n_support_ restriction)
        dummy_data = np.zeros((2, aggregated_model.support_vectors_.shape[1]))
        dummy_target = np.array([0, 1], dtype=int)
        aggregated_model.fit(dummy_data, dummy_target)

        # Replace the support vectors, dual coefficients, and intercept with the aggregated values
        aggregated_model.support_vectors_ = np.vstack(aggregated_support_vectors)
        aggregated_model.dual_coef_ = np.vstack([np.array(row) for row in aggregated_coefficients])
        aggregated_model.intercept_ = aggregated_intercept

        return aggregated_model, {"aggregated_support_vectors": aggregated_model.support_vectors_,
                                  "aggregated_coefficients": aggregated_model.dual_coef_,
                                  "aggregated_intercept": aggregated_model.intercept_}

    def lr_aggregate(self):
        # Initialize the aggregated coefficients and intercept
        aggregated_coefficients = np.zeros(self.models_parameters[0]["coefs"].shape)
        aggregated_intercept = 0

        # Combine coefficients and intercepts from each client's model
        for model in self.models_parameters:
            aggregated_coefficients += model["coefs"]
            aggregated_intercept += model["intercept"]

        # Average the coefficients and intercept
        aggregated_coefficients /= len(self.models_parameters)
        aggregated_intercept /= len(self.models_parameters)

        # Create a new logistic regression model with the same parameters as the original models
        aggregated_model = deepcopy(self.global_model_)
        aggregated_model.coef_ = aggregated_coefficients
        aggregated_model.intercept_ = aggregated_intercept

        # Fit the aggregated model with dummy data
        dummy_data = np.zeros((2, aggregated_model.coef_.shape[1]))
        dummy_target = np.array([0, 1], dtype=int)
        aggregated_model.fit(dummy_data, dummy_target)

        # Replace the coefficients and intercept with the aggregated values
        aggregated_model.coef_ = aggregated_coefficients
        aggregated_model.intercept_ = aggregated_intercept

        return aggregated_model, {"aggregated_coefficients": aggregated_model.coef_,
                                  "aggregated_intercept": aggregated_model.intercept_}

    def gnb_aggregate(self):
        # Initialize the aggregated class priors, means, and variances
        aggregated_class_priors = np.zeros(self.models_parameters[0]["class_priors"].shape)
        aggregated_theta = np.zeros(self.models_parameters[0]["theta"].shape)
        aggregated_sigma = np.zeros(self.models_parameters[0]["sigma"].shape)

        # Combine class priors, means, and variances from each client's model
        for model in self.models_parameters:
            aggregated_class_priors += model["class_priors"]
            aggregated_theta += model["theta"]
            aggregated_sigma += model["sigma"]

        # Average the class priors, means, and variances
        aggregated_class_priors /= len(self.models_parameters)
        aggregated_theta /= len(self.models_parameters)
        aggregated_sigma /= len(self.models_parameters)

        # Create a new Gaussian Naive Bayes model with the same parameters as the original models
        aggregated_model = deepcopy(self.global_model_)
        aggregated_model.set_params(**self.global_model_.get_params())

        # Set the aggregated class priors, means, and variances in the new model
        aggregated_model.class_prior_ = aggregated_class_priors
        aggregated_model.theta_ = aggregated_theta
        aggregated_model._sigma = aggregated_sigma

        # Compute the var_ attribute based on the aggregated _sigma attribute
        aggregated_model.var_ = np.copy(aggregated_model._sigma)
        aggregated_model.var_[aggregated_model.var_ < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny

        # Use the classes_ attribute from one of the individual models
        aggregated_model.classes_ = self.models_parameters[0]["classes"]

        return aggregated_model, {"aggregated_class_priors": aggregated_model.class_prior_,
                                  "aggregated_theta": aggregated_model.theta_,
                                  "aggregated_sigma": aggregated_model._sigma,
                                  "classes": aggregated_model.classes_}

    def sgd_aggregate(self):
        # Initialize the aggregated coefficients and intercept
        aggregated_coefficients = np.zeros(self.models_parameters[0]["coefs"].shape)
        aggregated_intercept = np.zeros(self.models_parameters[0]["intercept"].shape)

        # Combine coefficients and intercepts from each client's model
        for model in self.models_parameters:
            aggregated_coefficients += model["coefs"]
            aggregated_intercept += model["intercept"]

        # Average the coefficients and intercept
        aggregated_coefficients /= len(self.models_parameters)
        aggregated_intercept /= len(self.models_parameters)

        # Create a new SGDClassifier model with the same parameters as the original models
        aggregated_model = deepcopy(self.global_model_)
        aggregated_model.coef_ = aggregated_coefficients
        aggregated_model.intercept_ = aggregated_intercept

        aggregated_model.classes_ = self.models_parameters[0]["classes"]

        # print("\naggregated coefficients: ", aggregated_coefficients)
        # print("\naggregated intercept: ", aggregated_intercept)

        return aggregated_model, {"aggregated_coefficients": aggregated_model.coef_,
                                  "aggregated_intercept": aggregated_model.intercept_,
                                  "classes": aggregated_model.classes_}

    def mlp_aggregate(self):
        if len(self.models_parameters) == 0:
            raise ValueError("No models in the list")

            # Create a clone of the first model as the base for the aggregated model
        aggregated_model = deepcopy(self.global_model_)

        # Initialize lists to store the averaged weights and biases of all models
        averaged_coefs = [np.zeros_like(coef) for coef in self.models_parameters[0]["coefs"]]
        averaged_intercepts = [np.zeros_like(intercept) for intercept in self.models_parameters[0]["intercepts"]]

        # Calculate the average of the weights and biases from all models
        num_models = len(self.models_parameters)
        for model in self.models_parameters:
            for i, coef in enumerate(model["coefs"]):
                averaged_coefs[i] += coef / num_models
            for i, intercept in enumerate(model["intercepts"]):
                averaged_intercepts[i] += intercept / num_models

        # Set the averaged weights and biases for the aggregated model
        # Refit the model on a small dataset to update the internal structure
        dummy_data = np.zeros((2, self.models_parameters[0]["coefs"][0].shape[0]))
        dummy_target = np.array([0, 1], dtype=int)
        aggregated_model.fit(dummy_data, dummy_target)

        # Replace the weights and biases with the aggregated values
        aggregated_model.coefs_ = averaged_coefs
        aggregated_model.intercepts_ = averaged_intercepts

        return aggregated_model, {"aggregated_coefs": aggregated_model.coefs_,
                                  "aggregated_intercepts": aggregated_model.intercepts_}

    def svm_rbf_kernel(self, X1, X2, gamma):
        sq_dist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dist)

    def svm_linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def svm_poly_kernel(self, X1, X2, degree, gamma, coef0):
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree

    def svm_sigmoid_kernel(self, X1, X2, gamma, coef0):
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

    def svm_custom_decision_function(self, model, X):
        if model.kernel == 'linear':
            kernel_matrix = self.svm_linear_kernel(X, model.support_vectors_)
        elif model.kernel == 'poly':
            kernel_matrix = self.svm_poly_kernel(X, model.support_vectors_, model.degree, model.gamma, model.coef0)
        elif model.kernel == 'rbf':
            kernel_matrix = self.svm_rbf_kernel(X, model.support_vectors_, model.gamma)
        elif model.kernel == 'sigmoid':
            kernel_matrix = self.svm_sigmoid_kernel(X, model.support_vectors_, model.gamma, model.coef0)
        elif model.kernel == 'precomputed':
            raise NotImplementedError("Custom decision function doesn't support 'precomputed' kernel")
        else:
            raise ValueError("Unknown kernel type")

        decision_values = np.dot(kernel_matrix, model.dual_coef_.T) + model.intercept_
        return decision_values

    def svm_custom_predict(self, model, X):
        decision_values = self.svm_custom_decision_function(model, X)
        if len(model.classes_) == 2:
            y_pred = np.where(decision_values >= 0, model.classes_[1], model.classes_[0])
        else:
            y_pred = model.classes_[np.argmax(decision_values, axis=1)]

        return y_pred.ravel()

    def svm_summary(self, model):
        print("\tSupport Vector Machine Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "SVC"
        print(f"\t\t==> Kernel: {model.kernel}")

        if model.kernel == 'poly':
            print(f"\t\t==> Degree: {model.degree}")

        if model.kernel in ['poly', 'rbf', 'sigmoid']:
            print(f"\t\t==> Gamma: {'auto' if model.gamma == 'auto' else model._gamma}")

        if model.kernel in ['poly', 'sigmoid']:
            print(f"\t\t==> Coef0: {model.coef0}")

        print(f"\t\t==> C (Regularization parameter): {model.C}")
        print(f"\t\t==> Shrinking: {model.shrinking}")
        print(f"\t\t==> Probability estimates: {model.probability}")
        print(f"\t\t==> Tolerance: {model.tol}")
        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Number of support vectors: {model.n_support_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def lr_summary(self, model):
        print("\tLogistic Regression Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "LogisticRegression"
        print(f"\t\t==> Solver: {model.solver}")
        print(f"\t\t==> Penalty: {model.penalty}")
        print(f"\t\t==> C (Inverse regularization strength): {model.C}")
        print(f"\t\t==> Fit intercept: {model.fit_intercept}")
        print(f"\t\t==> Max iterations: {model.max_iter}")
        print(f"\t\t==> Tolerance: {model.tol}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Coefficients: {model.coef_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def gnb_summary(self, model):
        print("\tGaussian Naive Bayes Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "GaussianNB"
        print(f"\t\t==> Variance smoothing: {model.var_smoothing}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Class priors: {model.class_prior_}")
            print(f"\t\t==> Class counts: {model.class_count_}")
            print(f"\t\t==> Mean: {model.theta_}")
            print(f"\t\t==> Variance: {model.sigma_} \n\t=============================\n")
        except BaseException as e:
            pass

    def sgd_summary(self, model):
        print("\tSGD Classifier Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "SGDClassifier"
        print(f"\t\t==> Loss: {model.loss}")
        print(f"\t\t==> Penalty: {model.penalty}")
        print(f"\t\t==> Alpha: {model.alpha}")
        print(f"\t\t==> L1 ratio: {model.l1_ratio}")
        print(f"\t\t==> Fit intercept: {model.fit_intercept}")
        print(f"\t\t==> Max iterations: {model.max_iter}")
        print(f"\t\t==> Tolerance: {model.tol}")
        print(f"\t\t==> Learning rate: {model.learning_rate}")
        print(f"\t\t==> Eta0: {model.eta0}")

        try:
            print(f"\t\t==> Class labels: {model.classes_}")
            print(f"\t\t==> Coefficients: {model.coef_}")
            print(f"\t\t==> Intercept: {model.intercept_} \n\t=============================\n")
        except BaseException as e:
            pass

    def mlp_summary(self, model):
        print("\tMLP Model Summary\n\t=============================")
        print("\t\t==> Model type:", type(model).__name__)  # Output: "MLPClassifier" or "MLPRegressor"

        print("\t\t==> Activation function:", model.activation)
        print("\t\t==> Solver (optimizer):", model.solver)
        print("\t\t==> Alpha (L2 regularization):", model.alpha)
        print("\t\t==> Learning rate:", model.learning_rate)
        print("\t\t==> Initial learning rate (eta0):", model.learning_rate_init)

        # Print hidden layer sizes
        print("\t\t==> Hidden layer sizes:", model.hidden_layer_sizes)

        # Print weights and biases for each layer
        print("\t\t==> Layer weights and biases:")
        for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
            print(f"\t\t==> Layer {i + 1}:")
            print(f"\t\t\tWeights: {coef}")
            print(f"\t\t\tBiases: {intercept}")
        print("\n\t=============================\n")
