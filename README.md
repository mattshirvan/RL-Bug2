```
# RL bug2 algorithm for pathfinding

while the goal T is not achieved, do:
    begin
        while the path ST to the goal is not obstructed, do
            begin
                update the reward function based on the current state and action taken, including a penalty for any distance from the line ST
                select the best action to take based on the transition probabilities of the rewards
                move towards the goal along the path ST,
                if the path is obstructed then
                    begin
                        mark the current location as P 
                        circumnavigate the object until the robot either:
                        (a) hits the line ST at a point closer to T than P and can move towards T, in which case the robot follows ST;
                        (b) returns to where P in which case T is unreachable.
                    end
            end
    end
```
