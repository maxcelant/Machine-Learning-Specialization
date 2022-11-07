import numpy as np
import copy

# Recall that f(x) = w1x1 + w2x2 + w3x3 +...+ wnxn + b
# Thats what this function does, but for just one row at a time
# this is a VERY inefficient way of doing this
def predict_single_loop(x, w, b):
        n = x.shape[0] # number of features
        p = 0
        for i in range(n):
            p_i = x[i] * w[i]
            p = p + p_i
        p = p + b
        return p


# correct way of doing the above code^
def predict(x, w, b):
    return np.dot(w, x) + b


# J(w, b) aka Cost Function
def compute_cost(X, y, w, b):
    '''
    X (ndarray (m,n)): Data, m examples with n features
    y (ndarray (m,)) : target values
    w (ndarray (n,)) : model parameters  
    b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    '''
    m = X.shape[0] # number of training examples
    cost = 0.0
    for i in range(m):
        y_hat_i = np.dot(X[i], w) + b
        cost = cost + (y_hat_i - y[i])**2
    cost = cost / (2 * m)
    return cost


def compute_gradient(X, y, w, b):
    m, n = X.shape # (number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    for i in range(m):
        y_hat = (np.dot(X[i], w) + b)
        err = y_hat - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db
        

def gradient_descent(X, y, w_in, b_in, alpha, epochs):
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(epochs):
        dj_dw, dj_db = compute_gradient(X, y, w, b) 
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w, b


def main():
    #                        input features                                    output target
    # (Size (sqft), Number of Bedrooms, Number of Floors, Age of Home) -> Price (1000s of dollars)
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]]) # input features
    y_train = np.array([460, 232, 178]) # output targets

    alpha = 5.0e-7 # learning rate
    epochs = 1000  # number of iterations
    
    # w is a vector with 'n' elements, each corresponding to one feature
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
    
    # bias parameter
    b_init = 785.1811367994083
    
    # get the first row from the X Matrix
    x_vec = X_train[0,:] 
    y_hat = predict_single_loop(x_vec, w_init, b_init)
    
    # same thing, but more efficient and easier
    y_hat = predict(x_vec, w_init, b_init)
    
    # We can compute the cost error function like so
    cost = compute_cost(X_train, y_train, w_init, b_init)
    
    w_final, b_final = gradient_descent(X_train, y_train, w_init, b_init, )
    
    m, _ =  X_train.shape
    for i in range(m):
        print(f'Prediction: {np.dot(X_train[i], w_final) + b_final}, Target: {y_train[i]}')
    


if __name__ == '__main__':
    main()
