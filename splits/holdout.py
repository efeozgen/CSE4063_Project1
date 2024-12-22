from sklearn.model_selection import train_test_split

class HoldoutSplit:
    def __init__(self, test_size, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return [(X_train, X_test, y_train, y_test)]
    
    # def prt(self, x, y):
    #     print(x.head())
    #     print(y.head())