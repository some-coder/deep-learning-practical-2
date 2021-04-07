library(readr)
library(viridis)
library(reshape2)
library(ggplot2)
library(stringr)
library(dplyr)


#info <- read_file("github/deep-learning-practical-2/analysis/csv_info.txt")
files <- list.files("Downloads/DL_results/")
path = "Downloads/DL_results/"
i = 1
datalist = list()
for (file in files){
  file_path <- paste(path, file, sep="")
  if (grepl("rew", file, fixed=TRUE) == TRUE) { # could probably use regex for this but lazy
    
    temp_reward_data <- read_csv(file_path)
    
    agent_name <- str_replace_all(file, "rew-full_exp_", "")
    agent_name <- str_replace_all(agent_name,"\\d+", "")
    agent_name <- str_replace_all(agent_name, "_.csv", "")
    agent_name <- str_replace_all(agent_name, ".csv", "")
    
    temp_reward_data$agent <- agent_name 
    
    temp_reward_data$run <- str_extract(file, "\\d+")
    temp_reward_data$run[is.na(temp_reward_data$run)] <- 1
    
    agent_name <- str_replace_all(file, "rew-full_exp_", "")
    agent_name <- str_replace_all(agent_name, "_.csv", "")
    agent_name <- str_replace_all(agent_name, ".csv", "")
    temp_reward_data$agentRUN <- agent_name
    
    
    datalist[[i]] <- temp_reward_data
    
    i = i + 1
    
  }
}
df <- do.call(rbind, datalist)
View(df)

df$time_step
final_df <- df[which(df$time_step > 798990),]

View(final_df)



ggplot(final_df, aes(time_step, mean_reward)) + geom_line(aes(color=agentRUN)) + 
  ggtitle("Rewards over time") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()

ggplot(final_df, aes(time_step, mean_cost)) + geom_line(aes(color=agentRUN)) + 
  ggtitle("Cost over time") + 
  theme(plot.title = element_text(hjust = 0.5)) + theme_bw()

group_by(final_df, agentRUN) %>%
  summarise(
    count = n(),
    mean = mean(mean_reward, na.rm = TRUE),
    sd = sd(mean_reward, na.rm = TRUE)
)

group_by(final_df, agentRUN) %>%
  summarise(
    count = n(),
    mean = mean(mean_cost, na.rm = TRUE),
    sd = sd(mean_cost, na.rm = TRUE)
  )

resREW.aov <- aov(mean_reward ~ agentRUN, data = final_df)
# Summary of the analysis
summary(resREW.aov)
TukeyHSD(resREW.aov)


resCOST.aov <- aov(mean_cost ~ agentRUN, data = final_df)
# Summary of the analysis
summary(resCOST.aov)
TukeyHSD(resCOST.aov)
