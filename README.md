# RL-Bug2

While the goal T is not achieved, do:
\begin{algorithmic}
    \While{the path ST to the goal is not obstructed}
        \State Update the reward function based on the current state and action taken, including a penalty for any distance from the line ST
        \State Select the best action to take based on the transition probabilities of the rewards
        \State Move towards the goal along the path ST
        \If{the path is obstructed}
            \State Mark the current location as P 
            \State Circumnavigate the object until the robot either:
            \begin{enumerate}
                \item Hits the line ST at a point closer to T than P and can move towards T, in which case the robot follows ST
                \item Returns to where P in which case T is unreachable
            \end{enumerate}
        \EndIf
    \EndWhile
\end{algorithmic}
