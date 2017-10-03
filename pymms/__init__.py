"""
pymms module for Python
=========================

pymms is yet another python implementation. It provides a "pure"
(requiring only numpy) implementation of the Viterbi
algorithm. Because it is based on numpy, the performance is still good
while providing more insight than the C implementation of hmmlearn.

Also it provides a classifier based emission model as well as well as
a graphviz based visualization for the transition matrix.

The graphviz visualization requires to install graphviz and that dot
is on the PATH.

(c) 2017 Guillaume Coffin

Licensed under the GNU General Public Licence v3 (GPLv3)
https://www.gnu.org/licenses/gpl-3.0.txt
"""

__version__ = "0.1"
__author__ = "Guillaume Coffin <guill.coffin@gmail.com>"
__license__ = 'LGPL v3'
__copyright__ = 'Copyright 2017 Guillaume Coffin'

from .hmm import *
