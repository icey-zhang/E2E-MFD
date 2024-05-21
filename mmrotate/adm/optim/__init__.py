from .nash.balancer import NashMTL
from .aligned.balancer import AlignedMTLBalancer, AlignedMTLUBBalancer
from .cagrad.balancer import CAGradBalancer
from .graddrop.balancer import GradDropBalancer
from .gradnorm.balancer import GradNormBalancer
from .gradvac.balancer import GradVacBalancer
from .mgda.balancer import MGDABalancer, MGDAUBBalancer
from .pcgrad.balancer import PCGradBalancer
from .uncertainty.balancer import HomoscedasticUncertaintyBalancer
from .ls.balancer import LinearScalarization
from .si.balancer import ScaleInvariantLinearScalarization
from .rlw.balancer import RandomLossWeighting
from .dwa.balancer import DynamicWeightAveraging
from .imtl.balancer import IMTLG

from .balancers import get_method
