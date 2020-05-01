from __future__ import absolute_import

class prettyLoss(object):
    
    STYLE = {
        'green' : '\033[32m',
        'red'   : '\033[91m', 
        'bold'  : '\033[1m', 
    }
    STYLE_END = '\033[0m'
    
    def __init__(self, show_percentage=False):
        
        self.show_percentage = show_percentage
        self.color_up = 'green'
        self.color_down = 'red'
        self.loss_terms = {}
    
    def __call__(self, epoch=None, **kwargs):
        
        if epoch is not None:
            print_string = 'Epoch {: 5d} '.format(epoch)
        else:
            print_string = ''

        for key, value in kwargs.items():
            
            pre_value = self.loss_terms.get(key, value)
            
            if value > pre_value:
                indicator  = '▲'
                show_color = self.STYLE[self.color_up]
            elif value == pre_value:
                indicator  = ''
                show_color = ''
            else:
                indicator  = '▼'
                show_color = self.STYLE[self.color_down]
            
            if self.show_percentage:
                show_value = 0 if pre_value == 0 \
                             else (value - pre_value) / float(pre_value)
                key_string = '| {}: {}{:3.2f}({:+3.2%}) {}'.format(key,show_color,value,show_value,indicator)
            else: 
                key_string = '| {}: {}{:.4f} {}'.format(key,show_color,value,indicator)
            
            # Trim some long outputs
            key_string_part = key_string[:32]
            print_string += key_string_part+self.STYLE_END+'\t'
            
            self.loss_terms[key] = value
            
        print(print_string)