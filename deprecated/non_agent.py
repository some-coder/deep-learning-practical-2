
class Non_Agent(object):
    def __init__(self, maintenance_interval, dyke_1_m, dyke_2_n):
        self.maintenance_interval = maintenance_interval
        self.dyke_1_m = dyke_1_m
        self.dyke_2_n = dyke_2_n
        self.action_interval = 0

    def act(self, time):
        if time >= self.action_interval:
            actions = [1] * (self.dyke_1_m + self.dyke_2_n)
            self.action_interval = time + self.maintenance_interval
        else:
            actions = [0] * (self.dyke_1_m + self.dyke_2_n)
        return actions

    def observe(self, terminal, reward):
        return
