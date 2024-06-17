import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# import metrics to evaluate clustering models
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score


import itertools



class Experimentacion_modelo:
    ## Atributes
    def __init__(self, model, hiper_parms, x_train, y_train, x_test=None, k_fold=4):
        self.model = model
        self.hiperparametros = hiper_parms  # Diccionario de hiperparametros
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        # self.y_test = y_test
        self.k_fold = KFold(n_splits=k_fold, shuffle=True, random_state=42)

        # atributos para guardar resultados
        self.resultados = (
            list()
        )  # lista de resultados [acuracy, presicion, recall, f1] de cada combinacion de hiperparametros
        self.combinaciones_hiper = None  # Guardar combinaciones de hiperparametros

        # best hiperparametros
        self.best_hiper = None

    def set_combinaciones(self):
        combinaciones_hiper = []
        for tupla in itertools.product(*self.hiperparametros.values()):
            tipos_originales = [type(valor) for valor in self.hiperparametros.values()]
            nueva_tupla = [valor if isinstance(tipo, type) else [valor] for tipo, valor in zip(tipos_originales, tupla)]
            combinaciones_hiper.append(nueva_tupla)

        self.combinaciones_hiper = combinaciones_hiper
        return combinaciones_hiper

    ## Methods
    def experimentacion(self):
        # Generar combinaciones de hiperparametros a probar
        combinaciones_hiper = self.set_combinaciones()

        # Entrenar modelo y encontrar hiperparametros
        for hiper in combinaciones_hiper:
            result_temp = []
            # Splitear datos de entrenamiento en k partes (k-fold cross validation)
            for train_index, test_index in self.k_fold.split(self.x_train):
                x_train_k, y_train_k = (
                    self.x_train[train_index],
                    self.y_train[train_index],
                )
                x_test_k, y_test_k = self.x_train[test_index], self.y_train[test_index]

                # Entrenar modelo
                self.model.set_params(*hiper)
                self.model.train(x_train_k, y_train_k)

                # Evaluar modelo
                y_pred = self.model.predict(x_test_k)

                # Guardar resultados de los scores obtenidos
                result_temp.append(self.get_metrics_macro(y_test_k, y_pred))
                # la otra es usar precision_recall_curve

            # Sacar los promedios de cada score obtenido en cada fold
            result_temp = np.array(result_temp)
            result_temp = np.mean(result_temp, axis=0)
            self.resultados.append(result_temp)

        # Extraer los mejores hiperparametros
        self.best_hiper = self.extract_best_hiper()
        return {key: value for key, value in zip(self.hiperparametros.keys(), self.best_hiper[0])}

    def get_metrics(self, y_true, y_pred):
        # Calculate clustering metrics
        
        silhouette = silhouette_score(self.x_train, y_pred)
        db_index = davies_bouldin_score(self.x_train, y_pred)
        ch_index = calinski_harabasz_score(self.x_train, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        mi = mutual_info_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)



        return [silhouette, db_index, ch_index, ari, mi, ami]

    def extract_best_hiper(self):
        # Sacar el indice del mejor resultado
        # Sacar los hiperparametros que dieron el mejor resultado
        resultados_ordenados = sorted(enumerate(self.resultados), key=lambda x: (x[1][3],x[1][0]), reverse=True)
        # Extraer los índices de los 5 mejores
        mejores_indices = [indice for indice, _ in resultados_ordenados[:5]]
        best_hiper = [self.combinaciones_hiper[indice] for indice in mejores_indices]
        self.best_hiper = best_hiper
        return best_hiper

    # ploteo de resultados
    def get_matrix_similitud(self,):
        # TODO: Implementar
        pass

    
    def plot_metrics(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        # METRICS
        silhouette = resultados[:, 0]
        db_index = resultados[:, 1]
        ch_index = resultados[:, 2]
        ari = resultados[:, 3]
        mi = resultados[:, 4]
        ami = resultados[:, 5]



        # Crear una lista con los nombres de las combinaciones de hiperparámetros
        num_combinaciones = len(self.combinaciones_hiper)
        nombres_combinaciones = [f"C_Hyper_{i+1}" for i in range(num_combinaciones)]

        # Crear un DataFrame para facilitar el uso de Seaborn
        df = pd.DataFrame({
            "Combinaciones de Hiperparámetros": nombres_combinaciones * 6,
            "Metric": ["Silhouette"] * num_combinaciones + ["DB Index"] * num_combinaciones + ["CH Index"] * num_combinaciones + ["ARI"] * num_combinaciones + ["MI"] * num_combinaciones + ["AMI"] * num_combinaciones,
            "Score": np.concatenate([silhouette, db_index, ch_index, ari, mi, ami])
        })

        # Plotear
        fig, ax = plt.subplots(figsize=(10, 8))

        # Barras
        sns.barplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metrics", data=df, ax=ax, width=0.3,legend=True)
        # Dispersion
        # Gráfico de dispersión
        # sns.scatterplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metric", data=df, ax=ax, s=60, legend=False)
        # # Lines
        # sns.lineplot(x="Combinaciones de Hiperparámetros", y="Score", hue="Metric", data=df, ax=ax, linewidth=2)

        df_numeric = df.select_dtypes(include=['number'])

        # Aplicar min() solo a las columnas numéricas
        y_limit_inf = df_numeric.min().min()
        # y_limit_inf = min(df.min())

        ax.set_ylim(y_limit_inf, 1)
        ticks = np.arange(y_limit_inf, 1.01, 0.01)
        ax.set_yticks(ticks)
        ax.autoscale_view() # autosacalar ticks

        # Lineas horizontales en el graico
        ax.grid(True, axis="y", ls="-", color="gray", alpha=0.4)

        ax.set_xlabel("Combinaciones de hiperparámetros")
        ax.set_ylabel("Scores")
        ax.set_title("Scores vs Combinaciones de hiperparámetros")
        plt.xticks(rotation=75)
        plt.legend(framealpha=0.4, loc="upper right")
        plt.tight_layout()  # Ajustar automáticamente los márgenes de la figura
        plt.show()


    def generater_reporte(self):
        # Extraer los resultados de los scores
        resultados = np.array(self.resultados)
        # METRICS
        silhouette = resultados[:, 0]
        db_index = resultados[:, 1]
        ch_index = resultados[:, 2]
        ari = resultados[:, 3]
        mi = resultados[:, 4]



        
        # Generar reporte to clustering
        reporte = pd.DataFrame(
            {
                "Silhouette": silhouette,
                "DB Index": db_index,
                "CH Index": ch_index,
                "ARI": ari,
                "MI": mi,
            }
        )
        sorted_reporte = reporte.sort_values(by=["ARI", "Silhouette"], ascending=False)
        return sorted_reporte

    def training_model_direct(self, hiper):
        self.model.set_params(*hiper)
        self.model.train(self.x_train, self.y_train)

    # Entrenar modelo con los mejores hiperparametros para la predicción final
    def training_model(self, i=0):
        self.model.set_params(*self.best_hiper[i])
        self.model.train(self.x_train, self.y_train)

    # Entrenar modelo sin tener que esperar

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        return y_pred
    

