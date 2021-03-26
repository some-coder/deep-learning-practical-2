library(readr)
library(viridis)
library(reshape2)
library(ggplot2)

info <- read_file("github/deep-learning-practical-2/analysis/csv_info.txt")
states_data <- read_csv("github/deep-learning-practical-2/analysis/states_data.csv") #change to some path

states <- states_data[-c(23:42)]
actions <- states_data[-c(3:22)]
View(states)
View(actions)



states_long <- melt(data = states, 
                 id.vars = c("time", "reward"), 
                 variable.name = "dyke", 
                 value.name = "state"
)
head(states_long)

#states_long$broken <- ifelse(states_long$state >= 1, 10, states_long$state)



#actions
actions_long <- melt(data = actions, 
                    id.vars = c("time", "reward"), 
                    variable.name = "action", 
                    value.name = "actions"
)


heatmap_dykes <- ggplot(states_long, aes(time, dyke, fill=state )) + 
  geom_tile() + scale_fill_viridis(name = "deterioration level dyke", ) + 
  ggtitle("Deterioration of dyke segments over time") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()


heatmap_actions <- ggplot(actions_long, aes(time, action, fill=actions)) + 
  geom_tile() + scale_fill_viridis(option = "magma", name = "action") + 
  ggtitle("Actions of agent on dyke segments over time") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()



ggplot(states, aes(time, reward)) + geom_line() + 
  ggtitle("Rewards over time") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()

ggplot(states, aes(time, dyke_1_1)) +
  geom_line()+geom_hline(yintercept=1.00, colour='red') + 
  ggtitle("Deterioration of dyke 1_1") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()


heatmap_dykes
heatmap_actions
