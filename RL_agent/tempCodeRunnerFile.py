# BESS NIGHT CHARGING - Predictive Strategy
            # Reward BESS for intelligently pre-charging based on solar forecast
            bess_soc = self.soc[2]  # BESS SoC
            bess_charge_power = self.node_battery_power_kw[self.bess_index]  # Node 10
            
            if bess_charge_power < 0:  # BESS is charging
                # Look ahead to assess if this night charging is justified
                steps_remaining = self.max_steps - self.current_step
                steps_until_evening = min(steps_remaining, int((18 - hour) * 4))
                
                if steps_until_evening > 4:
                    future_solar = self.solar_episode[self.current_step:self.current_step + steps_until_evening]
                    future_load = self.load_episode[self.current_step:self.current_step + steps_until_evening]
                    expected_surplus = np.sum(future_solar) - np.sum(future_load)
                    
                    # Calculate if solar will be sufficient
                    expected_surplus_energy = expected_surplus * 0.25 * 0.95
                    energy_needed = (0.75 - bess_soc) * self.bess_cap
                    
                    if expected_surplus_energy < energy_needed:
                        # Solar insufficient - STRONG reward for smart night charging
                        night_charge_bonus += -18.0 * bess_charge_power
                    else:
                        # Solar will be sufficient - moderate reward (still economical at 13c)
                        night_charge_bonus += -8.0 * bess_charge_power
                else:
                    # Not enough lookahead data - reward based on SoC state
                    if bess_soc < 0.4:
                        night_charge_bonus += -15.0 * bess_charge_power
                    else:
                        night_charge_bonus += -10.0 * bess_charge_power
