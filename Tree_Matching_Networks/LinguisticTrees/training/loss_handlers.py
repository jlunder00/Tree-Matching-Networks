# Authored by: Jason Lunder, Github: https://github.com/jlunder00/

from .loss import SimilarityLoss, EntailmentLoss, InfoNCELoss, TextLevelBinaryLoss, TextLevelSimilarityLoss, TextLevelEntailmentLoss, TextLevelContrastiveLoss 

LOSS_HANDLERS = {
    'similarity': {
        'aggregative': TextLevelSimilarityLoss,
        'other': SimilarityLoss
    },
    'entailment': {
        'aggregative': TextLevelEntailmentLoss,
        'other': EntailmentLoss
    },
    'infonce': {
        'aggregative': TextLevelContrastiveLoss,
        'other': InfoNCELoss
    },
    'binary': {
        'aggregative': TextLevelBinaryLoss
    }
}

