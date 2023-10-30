from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .maiq_learner import MAIQLearner
from .maiq_learner_online import MAIQLearnerOnline
from .maddpg_learner import MADDPGLearner
from .sac_learner import SACLearner
from .ppo_learner import PPOLearner
from .maiq_continuous_learner import MAIQContinuousLearner
from .maiq_continuous_learner_online import MAIQContinuousLearnerOnline

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["maiq_learner"] = MAIQLearner
REGISTRY["maiq_learner_online"] = MAIQLearnerOnline
REGISTRY["maiq_continuous_learner"] = MAIQContinuousLearner
REGISTRY["maiq_continuous_learner_online"] = MAIQContinuousLearnerOnline
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["sac_learner"] = SACLearner
REGISTRY["ppo_learner"] = PPOLearner