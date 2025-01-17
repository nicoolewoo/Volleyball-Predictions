Machine Learning Models for Volleyball Gameplay Prediction

### Overview
This project applies machine learning to analyze volleyball gameplay dynamics. The primary goals were:
- Predict ball movement using linear regression.
- Predict player actions using a neural network.
- Address challenges in modeling volleyball dynamics, a less-analyzed sport compared to basketball or football.

### Datasets
The project utilized 4,830 annotated frames from 55 YouTube volleyball videos. Key features of the datasets include:
- **Ball Positions**: Pixel coordinates of the ball.
- **Player Coordinates**: Bounding box coordinates for each player.
- **Player Actions**: Labeled actions for 9 categories:
  - Waiting, Setting, Digging, Falling, Spiking, Blocking, Jumping, Moving, Standing.

#### Data Cleaning
1. **Ball Data**: Converted into a dataframe using `clean_data.py`, adding a `frame_id` column for alignment.
2. **Player Data**: Parsed player coordinates and actions into a dataframe using `dataset_detections` in `clean_data.py`.
3. **Merging**: Aligned datasets on `frame_id` and `frame_time` to unify ball and player data.

#### Challenges
- **Data Alignment**: Misalignment between ball and player datasets due to parsing issues.
- **Player Count Variance**: Some frames detected more than 6 players due to annotation inconsistencies.
- **Player Identification**: Attempted to track players across frames using Euclidean distance but faced limitations.

### Models and Methodology

#### 1. Linear Regression for Ball Trajectory
- **Features**: Ball \(x, y\) coordinates and velocities.
- **Target**: Predicted \(x, y\) positions of the ball.
- **Data Split**: Sequential 80% training and 20% testing split.
- **Results**:
  - For small datasets (e.g., 100 data points): Low RMSE and MSE.
  - For larger datasets: Model accuracy degraded due to increased complexity and nonlinear patterns.

#### 2. Neural Network for Player Actions
- **Features**: Ball \(x, y\) coordinates and player bounding box coordinates.
- **Target**: One-hot encoded action labels for 9 classes.
- **Architecture**:
  - Input: Fully connected layer with 64 neurons (ReLU activation).
  - Hidden: Dropout layer and a dense layer with 32 neurons.
  - Output: 9 neurons (Softmax activation).
- **Results**:
  - Achieved ~77% training accuracy but struggled with class imbalances.
  - Added regularization (L2) and class weights to improve predictions for underrepresented actions.

#### Key Observations
- The model frequently over-predicted "Standing" due to its dominance in the dataset.
- Rebalancing class weights improved predictions for rare actions but reduced overall accuracy.

### Validation
- Used metrics such as RMSE, MSE, MAE, and F1-score to evaluate model performance.
- Identified overfitting and class imbalance issues through confusion matrices and training logs.

### Challenges and Limitations
- Lack of \(z\)-axis data led to incomplete ball trajectory modeling.
- Inconsistent player counts across frames reduced player-specific tracking accuracy.
- Limited computational resources constrained model complexity.

### Future Work
- Incorporate \(z\)-axis data for better ball trajectory modeling.
- Use timestamps to isolate active plays from idle moments.
- Develop methods for robust player identification across frames.
- Experiment with deeper neural networks and advanced architectures to improve action predictions.

