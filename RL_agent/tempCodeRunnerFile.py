# 1. Common Solar Forecast (1 Value)
        if self.current_step < self.max_steps:
            # Use raw weather data from Node 0 as the 'signal'
            common_solar = np.array([self.solar_episode[self.current_step][0]])
            load_step = self.load_episode[self.current_step]
        else:
            # Safety: Return zeros if episode has ended
            common_solar = np.array([0.0])
            load_step = np.zeros(10)
        