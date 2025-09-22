import os
import cleaning.clean_all as ca
import cleaning.profile_pass as pp
print('Imports OK')
print('Has main:', hasattr(ca,'main'))
print('Profile stats exists:', os.path.exists('cleaning/profile_stats.json'))
