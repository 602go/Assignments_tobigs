import numpy as np
import math

class FourLayerNet():
    """
    4 Layer Network를 만드려고 합니다.

    해당 네트워크는 아래의 구조를 따릅니다.

    input - Linear - ReLU - Linear - ReLU - Linear - ReLU - Linear - Softmax

    Softmax 결과는 입력 N개의 데이터에 대해 개별 클래스에 대한 확률입니다.
    """

    def __init__(self, X, input_size, hidden1_size, hidden2_size, hidden3_size, output_size, std=1e-4):
         """
         네트워크에 필요한 가중치들을 initialization합니다.
         initialized by random values
         해당 가중치들은 self.params 라는 Dictionary에 담아둡니다.

         input_size: 데이터의 변수 개수 - D
         hidden_size: 히든 층의 H 개수 - H
         output_size: 클래스 개수 - C

         """

         self.params = {}
        #ReLU 활성화함수로 사용시 사용하는 가중치 초기화 방법
        # He initialization #2015년에 나온 비교적따끈따끈한 방법 #모두를위한딥러닝참고 #/np.sqrt(size/2)
        #Model4L에서 한층 더 쌓기(4 layers)
        #무작정 층을 더 쌓는건 성능 올리는데 좋은 방법이 아니라고 배웠다.
        #그러나 시도에 의의를 두겠다. #내가 언제 또 오류역전파를 직접 구현해보나 싶기 때문이다 
         self.params["W1"] = std * np.random.randn(input_size, hidden1_size)/np.sqrt(input_size/2)
         self.params["b1"] = np.random.randn(hidden1_size)
         self.params["W2"] = std * np.random.randn(hidden1_size, hidden2_size)/np.sqrt(hidden1_size/2)
         self.params["b2"] = np.random.randn(hidden2_size)
         self.params["W3"] = std * np.random.randn(hidden2_size, hidden3_size) / np.sqrt(hidden2_size / 2)
         self.params["b3"] = np.random.randn(hidden3_size)
         self.params["W4"] = std * np.random.randn(hidden3_size, output_size)/np.sqrt(hidden3_size/2)
         self.params["b4"] = np.random.randn(output_size)

    def forward(self, X, y=None):

        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        Linear - ReLU - Linear - ReLU - Linear - ReLU - Linear - Softmax - CrossEntropy Loss

        y가 주어지지 않으면 Softmax 결과 p와 Activation 결과 a,q,t를 return합니다. 모두 backward에서 미분할때 사용합니다.
        y가 주어지면 CrossEntropy Error를 return합니다.

        """

        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        N, D = X.shape

        # 여기에 p를 구하는 작업을 수행하세요.

        #Linear1
        h = np.dot(X,W1) + b1
        #ReLU1
        a = np.maximum(0,h)
        #Linear2
        o = np.dot(a,W2)+b2
        #ReLU2
        q = np.maximum(0,o)
        #Linear3
        z = np.dot(q,W3)+b3
        #ReLU3
        t = np.maximum(0,z)
        #Linear4
        m = np.dot(t,W4)+b4
        #softmax
        p = np.exp(m)/np.sum(np.exp(m),axis=1).reshape(-1,1)

        if y is None:
            return p, a, q, t
        
        # 여기에 Loss를 구하는 작업을 수행하세요.
        
        #CrossEntropy Loss (강의안 공식참고)
        Loss = 0
        for i in range(p.shape[0]):
            Loss -= np.log(p[i][y[i]])

        return Loss


    def backward(self, X, y, learning_rate=1e-5):
        """

        X: input 데이터 (N, D)
        y: 레이블 (N,)

        grads에는 Loss에 대한 W1, b1, W2, b2, W3, b3, W4, b4 미분 값이 기록됩니다.

        원래 backw 미분 결과를 return 하지만
        여기서는 Gradient Descent방식으로 가중치 갱신까지 합니다.

        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]
        W4, b4 = self.params["W4"], self.params["b4"]
        N = X.shape[0] # 데이터 개수
        grads = {}

        p, a, q, t = self.forward(X)

        # 여기에 파라미터에 대한 미분을 저장하세요.

        #softmax 미분
        #강의안에서 P-T에 해당
        dp = p
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if(j==y[i]):
                    dp[i][j]-=1
          
        #ReLU3 미분
        dt = np.heaviside(t,0)

        #ReLU2 미분
        dq = np.heaviside(q,0)
        
        #ReLU1 미분
        da = np.heaviside(a,0) #a가 0보다 크면 1, 작으면 0, 같으면 b(여기선 b가 0이니깐 오른쪽값 0)


        # 가중치와 바이어스 미분
        grads["W4"] = np.dot(t.T, dp)
        grads["b4"] = dp[0]
        
        grads["W3"] = np.dot(q.T, dt * np.dot(dp, W4.T))
        grads["b3"] = (dt * np.dot(dp, W4.T))[0]
        
        grads["W2"] = np.dot(a.T, dq * np.dot(dt * np.dot(dp, W4.T), W3.T))
        grads["b2"] = (dq * np.dot(dt * np.dot(dp, W4.T), W3.T))[0]
        
        #3 layers에서 새로 추가한 부분
        grads["W1"] = np.dot(X.T, da * np.dot(dq * np.dot(dt * np.dot(dp, W4.T), W3.T), W2.T))
        grads["b1"] = (da * np.dot(dq * np.dot(dt * np.dot(dp, W4.T), W3.T), W2.T))[0]
        

        #가중치와 바이어스 업데이트
        self.params["W4"] -= learning_rate * grads["W4"]
        self.params["b4"] -= learning_rate * grads["b4"]
        self.params["W3"] -= learning_rate * grads["W3"]
        self.params["b3"] -= learning_rate * grads["b3"]                
        self.params["W2"] -= learning_rate * grads["W2"]
        self.params["b2"] -= learning_rate * grads["b2"]
        self.params["W1"] -= learning_rate * grads["W1"]
        self.params["b1"] -= learning_rate * grads["b1"]

    def accuracy(self, X, y):

        p, r1, r2, r3 = self.forward(X)
        
        
        pre_p = np.argmax(p,axis=1)

        return np.sum(pre_p==y)/pre_p.shape[0]
