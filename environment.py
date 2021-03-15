import random

class Environment(object):
    def __init__(self,
                 m,
                 n,
                 alpha,
                 beta,
                 c_pm,
                 c_cm,
                 c_f,
                 c_s,
                 delta_t,
                 L
                ):
        """
        Constructs a new dyke repair reinforcement learning environment.

        @param m: The length of the first dyke.
        @param n: The length of the second dyke.
        @param alpha: The shape parameter for the Gamma process.
        @param beta: The scale parameter for the Gamma process.
        @param c_pm: The cost of preventing dyke breaches. Non-negative.
        @param c_cm: The cost of repairing dykes. Non-negative.
        @param c_f: A minimum cost of repair. Non-negative.
        @param c_s: The societal cost upon a dyke breach. Non-negative.
        @param delta_t: The timeframe on which to run the experiment. Positive.
        @param L: The deterioration level at which a dyke breaches. Non-negative.
        """
        self.m = m
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.c_pm = c_pm
        self.c_cm = c_cm
        self.c_f = c_f
        self.c_s = c_s
        self.delta_t = delta_t
        self.L = L

        self.state_1_x_t = [0] * self.m  # first dyke [X(t),...,]
        self.state_2_x_t = [0] * self.n  # second dyke [X(t),...,]
        self.state_1_t = [0] * self.m  # first dyke [t,...,]
        self.state_2_t = [0] * self.n  # second dyke [t,...,]

        self.actions_1 = [0]
        self.actions_2 = [0]

        # set seed
        self.random_generator = random.Random()
        self.random_generator.seed(1234)

    # functions for the agent
    def observe_state(self):
        return self.state_1_x_t + self.state_2_x_t

    def take_action(self, actions):
        self.actions_1 = actions[:len(self.state_1_x_t)]
        self.actions_2 = actions[len(self.state_1_x_t):]
        self.update_state()
        return True

    def get_reward(self):
        cost, cum_time = self.evaluate_state()
        return cost

    # internal functions
    def update_state(self):
        # first dyke update
        for i, X_t, in enumerate(self.state_1_x_t):
            action = self.actions_1[i]
            if action == 1:
                self.state_1_x_t[i] = 0
                self.state_1_t[i] = 0
            elif action == 0:
                self.state_1_x_t[i] += self.gamma_increment()
                self.state_1_t[i] += self.delta_t

        # second dyke update
        for i, X_t, in enumerate(self.state_2_x_t):
            action = self.actions_2[i]
            if action == 1:  # maintain
                self.state_2_x_t[i] = 0
                self.state_2_t[i] = 0
            elif action == 0:   # do nothing
                self.state_2_x_t[i] += self.gamma_increment()
                self.state_2_t[i] += self.delta_t
        return

    def evaluate_state(self):
        cost = 0
        time = 0

        nr_breaks_first_dyke = 0
        nr_breaks_second_dyke = 0

        maintenance = False

        # first dyke update
        for i, X_t, in enumerate(self.state_1_x_t):
            if X_t > self.L: # dyke breached
                maintenance = True
                nr_breaks_first_dyke += 1
                cost += self.c_cm
                time += self.state_1_t[i]
                self.state_1_x_t[i] = 0
                self.state_1_t[i] = 0
            else: # preventive maintenance
                if self.actions_1[i] == 1:
                    maintenance = True
                    cost += self.c_pm
                    time += self.state_1_t[i]
                    self.state_1_x_t[i] = 0
                    self.state_1_t[i] = 0
                # else --> do nothing

        # second dyke update
        for i, X_t, in enumerate(self.state_2_x_t):
            if X_t > self.L: # dyke breached
                maintenance = True
                nr_breaks_second_dyke += 1
                cost += self.c_cm
                time += self.state_2_t[i]
                self.state_2_x_t[i] = 0
                self.state_2_t[i] = 0
            else: # preventive maintenance
                if self.actions_2[i] == 1:
                    maintenance = True
                    cost += self.c_pm
                    time += self.state_2_t[i]
                    self.state_2_x_t[i] = 0
                    self.state_2_t[i] = 0
                # else --> do nothing

        if maintenance: # pay fixed cost
            cost += self.c_f

        if nr_breaks_first_dyke > 0 and nr_breaks_second_dyke > 0: # pay societal cost
            cost += self.c_s
        return cost, time

    def gamma_increment(self):
        return self.random_generator.gammavariate(alpha=self.alpha * self.delta_t,beta=self.beta)

