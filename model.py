from mesa import Model, Agent
from mesa.time import SimultaneousActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector


class VehicleAgent(Agent):
    """
    Vehicle agent
    """

    def __init__(self, pos, model, max_speed):
        """
        Create a new vehicle agent.
        Args:
           pos: Agent initial position in x, y.
           model: The model the agent is associated with.
           max_speed: The maximum number of cells an agent can move in a single step
        """
        super().__init__(pos, model)
        self.pos = pos
        self.speed = 0
        self.max_speed = max_speed
        self._next_pos = None

    def step(self):
        """
        Calculates the next position of the agent based on several factors:
        - Current Speed
        - Max Speed
        - Proximity of agent ahead of it
        - Random chance of deceleration
        """
        # STEP 1: ACCELERATION
        if self.speed < self.max_speed:
            self.speed += 1

        # STEP 2: DECELERATION
        distance_to_next = 0
        (x, y) = self.pos
        for distance in range(self.max_speed):
            distance += 1
            test_x = x + distance
            test_pos = self.model.grid.torus_adj((test_x, y))
            if self.model.grid.is_cell_empty(test_pos):
                distance_to_next += 1
                if distance_to_next == self.speed:
                    break
            else:
                break
        self.speed = distance_to_next

        # STEP 3: RANDOMISATION
        if self.random.random() < 0.3 and self.speed > 0:
            self.speed -= 1

        # STEP 4: MOVEMENT
        self._next_pos = self.pos
        (x, y) = self._next_pos
        x += self.speed
        self._next_pos = self.model.grid.torus_adj((x, y))

        self.model.total_speed = self.model.total_speed + self.speed

    def advance(self):
        """
        Moves the agent to its next position.
        """
        self.model.grid.move_agent(self, self._next_pos)


class NaSchTraffic(Model):
    """
    Model class for the Nagel and Schreckenberg traffic model.
    """

    def __init__(self, height=1, width=60, vehicle_quantity=5, general_max_speed=4, seed=None):
        """"""

        super().__init__(seed=seed)
        self.height = height
        self.width = width
        self.vehicle_quantity = vehicle_quantity
        self.general_max_speed = general_max_speed
        self.schedule = SimultaneousActivation(self)
        self.grid = SingleGrid(width, height, torus=True)

        self.average_speed = 0.0
        self.averages = []
        self.total_speed = 0

        self.datacollector = DataCollector(
            model_reporters={"Average_Speed": "average_speed"},  # Model-level count of average speed of all agents
            # For testing purposes, agent's individual x position and speed
            agent_reporters={
                "PosX": lambda x: x.pos[0],
                "Speed": lambda x: x.speed,
            },
        )

        # Set up agents
        # We use a grid iterator that returns
        # the coordinates of a cell as well as
        # its contents. (coord_iter)
        cells = list(self.grid.coord_iter())
        self.random.shuffle(cells)
        for vehicle_iter in range(0, self.vehicle_quantity):
            cell = cells[vehicle_iter]
            (content, x, y) = cell
            agent = VehicleAgent((x, y), self, general_max_speed)
            self.grid.position_agent(agent, (x, y))
            self.schedule.add(agent)

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """
        Run one step of the model. Calculate current average speed of all agents.
        """
        if self.schedule.steps == 100:
            self.running = False
        self.total_speed = 0
        # Step all agents, then advance all agents
        self.schedule.step()
        if self.schedule.get_agent_count() > 0:
            self.average_speed = self.total_speed / self.schedule.get_agent_count()
        else:
            self.average_speed = 0
        self.averages.append(self.average_speed)
        # collect data
        self.datacollector.collect(self)
