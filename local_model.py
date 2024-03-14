from sklearn import metrics
from copy import deepcopy
import numpy as np


class local_model:

    def __init__(self, model):
        self.model = model

    def evaluate(self, predicted, y_test):
        # evaluate model and print performance metrics
        print("\t<===> Model Performance Metrics: \n\t=============================")
        print("\t\t==> Accuracy: ", round(metrics.accuracy_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Precision: ", round(metrics.precision_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Recall: ", round(metrics.recall_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> F1 Score: ", round(metrics.f1_score(y_test, predicted) * 100, 2), "%")
        print("\t\t==> Specificity: ", round(metrics.recall_score
                                             (y_test, predicted, pos_label=0) * 100, 2), "%")
        print("\t\t==> Negative Predictive Value (NPV): ",
              round(metrics.precision_score(y_test, predicted, pos_label=0) * 100, 2), "%")

    def get_parameters(self):
        model_type = type(self.model).__name__
        if model_type == "SVC":
            parameters = self.svm_parameters()
        elif model_type == "LogisticRegression":
            parameters = self.lr_parameters()
        elif model_type == "GaussianNB":
            parameters = self.gnb_parameters()
        elif model_type == "SGDClassifier":
            parameters = self.sgd_parameters()
        elif model_type == "MLPClassifier":
            parameters = self.mlp_parameters()

        else:
            print("The provided model is not supported yet with this framework")
            return

        return parameters

    def set_parameters(self, parameters):
        model_type = type(self.model).__name__
        new_model = deepcopy(self.model)
        if model_type == "SVC":
            new_model.support_vectors_ = np.vstack(parameters["aggregated_support_vectors"])
            new_model.dual_coef_ = np.vstack([np.array(row) for row in parameters["aggregated_coefficients"]])
            new_model.intercept_ = parameters["aggregated_intercept"]
        elif model_type == "LogisticRegression":
            new_model.coef_ = parameters["aggregated_coefficients"]
            new_model.intercept_ = parameters["aggregated_intercept"]
        elif model_type == "GaussianNB":
            new_model.set_params(**self.model.get_params())
            # Set the aggregated class priors, means, and variances in the new model
            new_model.class_prior_ = parameters["aggregated_class_priors"]
            new_model.theta_ = parameters["aggregated_theta"]
            new_model._sigma = parameters["aggregated_sigma"]
            # Compute the var_ attribute based on the aggregated _sigma attribute
            new_model.var_ = np.copy(new_model._sigma)
            new_model.var_[new_model.var_ < np.finfo(np.float64).tiny] = np.finfo(np.float64).tiny
            # Use the classes_ attribute from one of the individual models
            new_model.classes_ = parameters["classes"]
        elif model_type == "SGDClassifier":
            new_model.coef_ = parameters["aggregated_coefficients"]
            new_model.intercept_ = parameters["aggregated_intercept"]
            new_model.classes_ = parameters["classes"]
        elif model_type == "MLPClassifier":
            new_model.coefs_ = parameters["aggregated_coefs"]
            new_model.intercepts_ = parameters["aggregated_intercepts"]

        return new_model

    def svm_parameters(self):
        support_vector = self.model.support_vectors_
        coefs = self.model.dual_coef_
        intercept = self.model.intercept_
        
        return {"support_vector": support_vector, "coefs": coefs, "intercept": intercept}

    def lr_parameters(self):
        coefs = self.model.coef_
        intercept = self.model.intercept_

        return {"coefs": coefs, "intercept": intercept}

    def gnb_parameters(self):
        class_priors = self.model.class_prior_
        theta = self.model.theta_
        sigma = self.model.sigma_
        classes = self.model.classes_

        return {"class_priors": class_priors, "theta": theta, "sigma": sigma, "classes": classes}

    def sgd_parameters(self):
        coefs = self.model.coef_
        intercept = self.model.intercept_
        classes = self.model.classes_

        #print("\ncoefs: ", coefs)
        #print("\nintercept: ", intercept)

        return {"coefs": coefs, "intercept": intercept, "classes": classes}

    def mlp_parameters(self):
        coefs = self.model.coefs_
        intercepts = self.model.intercepts_

        return {"coefs": coefs, "intercepts": intercepts}
