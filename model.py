import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os



class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    # actual prediction 
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # comment enregistré la data
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name )

class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimiser = optim.Adam(model.parameters(), self.learningRate)
        self.criterion = nn.MSELoss()


    # "gradient descent like for the prediction"
    def train_step(self, state, action, reward, next_state, game_over_state):
        # creation of the pytorch tensor :
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n,x) vector
        if len(state.shape) == 1:
            # only if we have a vector with one imnput (1)
            state = torch.unsqueeze(state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            next_state = torch.unsqueeze(next_state,0)
            game_over_state = (game_over_state, )

        # in first place we need predicted Q for state0 from our bellman equation 
        predicted = self.model(state)

        # we need apply the update rule only is the game_over_state if false
        # new prediction of Q is equal to Reward + gamma* maximal value of next predicted(state1)
        
        #but we have 3 value on the input so we clone the prediction 
        #predicted.clone()
        #predicted[argmax(action)] = new_predicted
        #new_predicted = + self.gamma*max()

        target = predicted.clone()
        for idx in range(len(game_over_state)):

            new_predicted = reward[idx]
            if not game_over_state[idx]:
                new_predicted = reward[idx] + self.gamma*torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()]  = new_predicted
        # set up the gradient to zero 
        self.optimiser.zero_grad()
        loss = self.criterion(target, predicted)
        #upgrade of the gradient
        loss.backward()

        self.optimiser.step()



