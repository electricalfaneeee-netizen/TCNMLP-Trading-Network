Another iteration of my attempts to make a profitable day trading bot.
This time I'm going to use a temporal convolutional network as the encoder, and some MLP layers with actor/critic logic (PPO) as the agent.
Thinking about another approach. Instead of an agent that trades like a human, I'm going to try creating some kind of predictor model that provides information relating to the future. If this works, deep learning might not even be required as the head for this. Something like XGBoost might even be sufficient.
