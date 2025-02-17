
def profiles(p_, f1, f2, f3, f4, t):
    if p_ == 'cold':
        if 0 <= t <= 0.5:
            if f1 == 1:
                return 16.0
            elif f1 == 2:
                return 7.5
            elif f1 == 3:
                return 3.75
            else: return 0.0
        if 0.5 < t <= 1.0:
            if f2 == 1:
                return 16.0
            elif f2 == 2:
                return 7.5
            elif f2 == 3:
                return 3.75
            else: return 0.0
        if 1.0 < t <= 1.5:
            if f3 == 1:
                return 16.0
            elif f3 == 2:
                return 7.5
            elif f3 == 3:
                return 3.75
            else: return 0.0
        if 1.5 < t <= 2.0:
            if f4 == 1:
                return 16.0
            elif f4 == 2:
                return 7.5
            elif f4 == 3:
                return 3.75
            else: return 0.0
                    
    if p_ == 'hot':
        if 0 <= t <= 0.5:
            if f1 == 1:
                return 16.0
            elif f1 == 2:
                return 7.5
            elif f1 == 3:
                return 3.75
            else: return 0.0
        if 0.5 < t <= 1.0:
            if f2 == 1:
                return 16.0
            elif f2 == 2:
                return 7.5
            elif f2 == 3:
                return 3.75
            else: return 0.0
        if 1.0 < t <= 1.5:
            if f3 == 1:
                return 16.0
            elif f3 == 2:
                return 7.5
            elif f3 == 3:
                return 3.75
            else: return 0.0
        if 1.5 < t <= 2.0:
            if f4 == 1:
                return 16.0
            elif f4 == 2:
                return 7.5
            elif f4 == 3:
                return 3.75
            else: return 0.0
        
            
    if p_ == 'd_prof':
        if 0 <= t <= 0.5:
            if f1 == 1:
                return 0.5
            elif f1 == 2:
                return 0.3
            elif f1 == 3:
                return 0.08
            else: return 0.0
        if 0.5 < t <= 1.0:
            if f2 == 1:
                return 0.5
            elif f2 == 2:
                return 0.3
            elif f2 == 3:
                return 0.08
            else: return 0.0
        if 1.0 < t <= 1.5:
            if f3 == 1:
                return 0.5
            elif f3 == 2:
                return 0.3
            elif f3 == 3:
                return 0.08
            else: return 0.0
        if 1.5 < t <= 2.0:
            if f4 == 1:
                return 0.5
            elif f4 == 2:
                return 0.3
            elif f4 == 3:
                return 0.08
            else: return 0.0
