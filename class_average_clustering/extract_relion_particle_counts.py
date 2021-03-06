from pathlib import Path
from helper_functions import save_obj, sort_dict


def gen_particle_counts(star_file):

    def convertToArrayWithSpaces(line):
        # return list of line values with spaces
        # ADD SPACES IN BETWEEN WHEN JOINING!
        l=line.rstrip().split(' ')
        newl = []
        val = ""
        for v in l:
                if v!="":
                        val+=v
                        newl.append(val)
                        val=""
                else: val+=" "
        return newl

    def get_class_col_num(filename):

        with open(filename) as file:    
            lines = file.readlines()
            for line in lines:
                if '_rlnClassNumber' in line:
                    class_col_num = int(line.split()[1][1:])
                    break

        return(class_col_num)


    particle_count_filename = star_file.replace('.star', '_particle_counts.pkl')

    if Path(particle_count_filename).exists() == False:

        print('saving %s' % particle_count_filename)
     
        class_col_num = get_class_col_num(star_file)
        
        particle_count_map = {}

        with open(star_file) as file:    
            lines = file.readlines()
            for line in lines:
                skip_line = False
                if line.startswith("data_particles") or line.startswith("data_\n") or ('opticsGroup' in line): skip_line = True
                l = line.strip().split()
                if skip_line or (len(l) <= 7): 
                    continue
                else:
                    line_list = convertToArrayWithSpaces(line)
                    class_num = int(line_list[class_col_num-1].strip())-1
                   
                    if class_num in particle_count_map:
                        particle_count_map[class_num] += 1
                    else:
                        particle_count_map[class_num] = 1

        particle_count_map = sort_dict(particle_count_map)
        print(particle_count_map)

        save_obj(particle_count_map, particle_count_filename) 
    
    return(particle_count_filename)

##
##
##

'''if _name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--star_file", help="*.star file [metadata generated by RELION or CryoSPARC .cs file converted .star via pyem (https://github.com/asarnow/pyem)]")
    args = parser.parse_args()

    if '.star' not in args.star_file:
        print('file must have extensions .star')
        sys.exit()

    get_particle_counts(args.star_file)'''
        




