import json
import numpy as np
import commands
import FaceDetection
import image_matching
import os

def labeling_user(input_js_filepath, output_js_filepath):
    with open(input_js_filepath) as input_f:
        label_count = 1
        for line in input_f:
            info_js = json.loads(line)
            info_js['label'] = label_count
            output_f = open(output_js_filepath, 'a')
            json.dump(info_js, output_f)
            output_f.write('\n')
            output_f.close()
            label_count += 1

def decode_twitter_short_url(input_js_filepath, output_js_filepath):
    found_last_time = False
    name = 'C_sharpxx'
    with open(input_js_filepath) as input_f:
        for line in input_f:
            info_js = json.loads(line)
            if info_js['screen_name'] == name:
                found_last_time = True
            if found_last_time:
                try:
                    info_js['personal_website'] = \
                        commands.getstatusoutput('curl -s -o /dev/null --head -w "%{url_effective}\n" -L "' +
                                                 info_js['personal_website'] + '"')[1]
                    output_f = open(output_js_filepath, 'a')
                    json.dump(info_js, output_f)
                    output_f.write('\n')
                    output_f.close()
                    print 'Decoding for user ' + info_js['screen_name']
                except KeyError:
                    output_f = open(output_js_filepath, 'a')
                    json.dump(info_js, output_f)
                    output_f.write('\n')
                    output_f.close()
                except UnicodeEncodeError:
                    output_f = open(output_js_filepath, 'a')
                    json.dump(info_js, output_f)
                    output_f.write('\n')
                    output_f.close()

                except UnicodeDecodeError:
                    output_f = open(output_js_filepath, 'a')
                    json.dump(info_js, output_f)
                    output_f.write('\n')
                    output_f.close()

def longestSubstringFinder(data):
    substr = ''
    if len(data) > 1 and len(data[0]) > 0:
        for i in range(len(data[0])):
            for j in range(len(data[0])-i+1):
                if j > len(substr) and is_substr(data[0][i:i+j], data):
                    substr = data[0][i:i+j]
    return substr

def is_substr(find, data):
    if len(data) < 1 and len(find) < 1:
        return False
    for i in range(len(data)):
        if find not in data[i]:
            return False
    return True

'''Threash is a percentage of how many eigen values kept'''
def calc_stringlist_similarity_LSA(string_list1, string_list2):
    sim_vector = []
    for term1 in string_list1:
        frequency_term1toterm2 = 0
        for term2 in string_list2:
            if term1 == term2:
                frequency_term1toterm2 += 1

        sim_vector.append(frequency_term1toterm2)

    sim_vector = np.array(sim_vector)[np.newaxis]
    frequency_matrix = np.dot(sim_vector.T, sim_vector)

    # svd decomposition
    U, s, V = np.linalg.svd(frequency_matrix, full_matrices=True)


    numerator_sum = 0

    for i in range(len(s)):
        numerator_sum += s[i] * s[i]

    denominator_sum = float(len(string_list1) * len(string_list2))

    try:
        sim_score = np.math.sqrt(numerator_sum)/np.math.sqrt(denominator_sum)

    except ZeroDivisionError:
        sim_score = 0

    if sim_score > 1:
        sim_score = 1

    return sim_score

def personal_website_similarity(twitter_js_filepath, flickr_js_filepath, save_txt_path):

    twi_js_file_linenum = sum(1 for line in open(twitter_js_filepath))
    flickr_js_file_linenum = sum(1 for line in open(flickr_js_filepath))

    row_index = 0
    col_index = 0

    '''initilize simlarity matrix'''
    sim_matrix = np.zeros((twi_js_file_linenum, flickr_js_file_linenum))

    twi_f = open(twitter_js_filepath)
    flickr_f = open(flickr_js_filepath)

    twi_lines = []
    flickr_lines = []

    for twi_line in twi_f:
        twi_lines.append(twi_line)

    for flickr_line in flickr_f:
        flickr_lines.append(flickr_line)

    for twi_line in twi_lines:
        for flickr_line in flickr_lines:

            try:
                twi_info_js = json.loads(twi_line)
                flickr_info_js = json.loads(flickr_line)
            except ValueError:
                continue

            try:
                print 'comparing the personal websites for %s and %s.' % \
                        (twi_info_js['screen_name'], flickr_info_js['ownername'])
                '''Change shortened Twitter URL to common URL'''
                real_twi_url = twi_info_js['personal_website'].lower().replace('http://', '').replace('https://', '').replace(' ', '')

                real_flckr_url = flickr_info_js['website'].lower().replace('http://', '').replace('https://', '').replace(' ', '')

                if len(real_twi_url) >= len(real_flckr_url):
                    short_url = real_flckr_url
                    long_url = real_twi_url

                else:
                    short_url = real_twi_url
                    long_url = real_flckr_url

                if short_url in long_url:
                    print real_flckr_url
                    print real_twi_url
                    print 'These two url are Same'
                    sim_matrix[row_index][col_index] = 1

            except KeyError:
                sim_matrix[row_index][col_index] = 0

            col_index += 1
        col_index = 0
        row_index += 1

    # saving matrix
    # later use np.loadtxt(txt_path) to load the variable
    print 'Finished comparing all the usernames, saving the similarity_matrix.'
    np.savetxt(save_txt_path, sim_matrix)
    return sim_matrix


def username_similarity(twitter_js_filepath, flickr_js_filepath, save_txt_path):
    twi_js_file_linenum = sum(1 for line in open(twitter_js_filepath))
    flickr_js_file_linenum = sum(1 for line in open(flickr_js_filepath))

    row_index = 0
    col_index = 0

    '''initilize simlarity matrix'''
    sim_matrix = np.zeros((twi_js_file_linenum, flickr_js_file_linenum))

    twi_f = open(twitter_js_filepath)
    flickr_f = open(flickr_js_filepath)

    twi_lines = []
    flickr_lines = []

    for twi_line in twi_f:
        twi_lines.append(twi_line)

    for flickr_line in flickr_f:
        flickr_lines.append(flickr_line)


    for twi_line in twi_lines:
        for flickr_line in flickr_lines:
            try:
                twi_info_js = json.loads(twi_line)
                flickr_info_js = json.loads(flickr_line)
            except ValueError:
                continue

            flickr_realname = None
            twi_name = twi_info_js['screen_name'].lower().replace(' ', '')
            flickr_ownername = flickr_info_js['ownername'].lower().replace(' ', '')

            print 'comparing %s and %s.' % (twi_name, flickr_ownername)

            try:
                flickr_realname = flickr_info_js['realname'].lower().replace(' ', '')
                flickr_realname = flickr_realname.lower().replace(' ', '')
            except KeyError:
                pass

            if flickr_realname is None:
                username_lcs = longestSubstringFinder([twi_name, flickr_ownername])
                sim_score = min(float(len(username_lcs))/float(len(twi_name)),
                                float(len(username_lcs)) / float(len(flickr_ownername)))

            else:
                username_lcs1 = longestSubstringFinder([twi_name, flickr_ownername])
                username_lcs2 = longestSubstringFinder([twi_name, flickr_realname])

                sim_score1 = min(float(len(username_lcs1))/float(len(twi_name)),
                                float(len(username_lcs1)) / float(len(flickr_ownername)))

                try:
                    sim_score2 = min(float(len(username_lcs2)) / float(len(twi_name)),
                                     float(len(username_lcs2)) / float(len(flickr_realname)))
                except ZeroDivisionError:
                    sim_score2 = 0

                sim_score = max(sim_score1, sim_score2)

            sim_matrix[row_index][col_index] = sim_score
            col_index += 1

        col_index = 0
        row_index += 1

    # saving matrix
    # later use np.loadtxt(txt_path) to load the
    print 'Finished comparing all the usernames, saving the similarity_matrix.'
    np.savetxt(save_txt_path, sim_matrix)
    return sim_matrix

def bio_similarity(twitter_js_filepath, flickr_js_filepath, save_txt_path):
    twi_js_file_linenum = sum(1 for line in open(twitter_js_filepath))
    flickr_js_file_linenum = sum(1 for line in open(flickr_js_filepath))

    row_index = 0
    col_index = 0

    '''initilize simlarity matrix'''
    sim_matrix = np.zeros((twi_js_file_linenum, flickr_js_file_linenum))

    twi_f = open(twitter_js_filepath)
    flickr_f = open(flickr_js_filepath)

    twi_lines = []
    flickr_lines = []

    for twi_line in twi_f:
        twi_lines.append(twi_line)

    for flickr_line in flickr_f:
        flickr_lines.append(flickr_line)

    for twi_line in twi_lines:
        for flickr_line in flickr_lines:

            try:
                twi_info_js = json.loads(twi_line)
                flickr_info_js = json.loads(flickr_line)
            except ValueError:
                continue

            try:
                twi_bio = twi_info_js['bio'].lower()
                flickr_bio = flickr_info_js['bio'].lower()

                twi_bio_terms = twi_bio.split(' ')
                flickr_bio_terms = flickr_bio.split(' ')

                if '' in twi_bio_terms:
                    twi_bio_terms.remove('')

                if '' in flickr_bio_terms:
                    flickr_bio_terms.remove('')

                if len(twi_bio_terms) == 0 or len(flickr_bio_terms) == 0:
                    sim_matrix[row_index][col_index] = 0

                # implement the LSA algorithm here
                else:
                    sim_matrix[row_index][col_index] = \
                        calc_stringlist_similarity_LSA(twi_bio_terms, flickr_bio_terms)


            except KeyError:
                sim_matrix[row_index][col_index] = 0

            col_index += 1

        col_index = 0
        row_index += 1

    # saving matrix
    # later use np.loadtxt(txt_path) to load the
    print 'Finished comparing all the bios, saving the similarity_matrix.'
    np.savetxt(save_txt_path, sim_matrix)
    return sim_matrix

def location_similaritydef(twitter_js_filepath, flickr_js_filepath, save_txt_path):
    twi_js_file_linenum = sum(1 for line in open(twitter_js_filepath))
    flickr_js_file_linenum = sum(1 for line in open(flickr_js_filepath))

    row_index = 0
    col_index = 0

    '''initilize simlarity matrix'''
    sim_matrix = np.zeros((twi_js_file_linenum, flickr_js_file_linenum))

    twi_f = open(twitter_js_filepath)
    flickr_f = open(flickr_js_filepath)

    twi_lines = []
    flickr_lines = []

    for twi_line in twi_f:
        twi_lines.append(twi_line)

    for flickr_line in flickr_f:
        flickr_lines.append(flickr_line)

    for twi_line in twi_lines:
        for flickr_line in flickr_lines:

            try:
                twi_info_js = json.loads(twi_line)
                flickr_info_js = json.loads(flickr_line)
            except ValueError:
                continue

            twi_location = None
            flickr_hometown = None
            flickr_current = None
            no_hometown = False
            no_current = False

            try:
                twi_location = twi_info_js['location']

            except KeyError:
                sim_matrix[row_index][col_index] = 0.0
                col_index += 1
                continue

            try:
                flickr_hometown = flickr_info_js['hometown']
            except KeyError:
                no_hometown = True

            try:
                flickr_current = flickr_info_js['currently']
            except KeyError:
                no_current = True

            if no_hometown and no_current:
                sim_matrix[row_index][col_index] = 0.0
                col_index += 1
                continue

            if twi_location is not None:
                twi_location = twi_location.lower().replace(' ', '').replace(',', '').replace('.', '')

            if flickr_hometown is not None:
                flickr_hometown = flickr_hometown.lower().replace(' ', '').replace(',', '').replace('.', '')

            if flickr_current is not None:
                flickr_current = flickr_current.lower().replace(' ', '').replace(',', '').replace('.', '')

            if twi_location == '':
                sim_matrix[row_index][col_index] = 0.0
                col_index += 1
                continue

            if flickr_hometown == '' and flickr_current == '':
                sim_matrix[row_index][col_index] = 0.0
                col_index += 1
                continue

            if flickr_current != '' and flickr_current is not None:
                if len(flickr_current) >= len(twi_location) and twi_location in flickr_current:
                    sim_matrix[row_index][col_index] = 1.0

                elif len(flickr_current) < len(twi_location) and flickr_current in twi_location:
                    sim_matrix[row_index][col_index] = 1.0

            elif sim_matrix[row_index][col_index] < 1 and flickr_hometown != '' and flickr_hometown is not None:
                if len(flickr_hometown) >= len(twi_location) and twi_location in flickr_hometown:
                    sim_matrix[row_index][col_index] = 1.0

                elif len(flickr_hometown) < len(twi_location) and flickr_hometown in twi_location:
                    sim_matrix[row_index][col_index] = 1.0


            col_index += 1

        col_index = 0
        row_index += 1

    # saving matrix
    # later use np.loadtxt(txt_path) to load the
    print 'Finished comparing all the bios, saving the similarity_matrix.'
    np.savetxt(save_txt_path, sim_matrix)
    return sim_matrix

'''call this function first before calling face recognition'''
def profile_img_similarity_no_face(twitter_js_filepath, flickr_js_filepath, twi_images_root_dir, flickr_images_root_dir,
                                   save_txt_path, both_face_txt_path):
    twi_js_file_linenum = sum(1 for line in open(twitter_js_filepath))
    flickr_js_file_linenum = sum(1 for line in open(flickr_js_filepath))

    cascPath = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml'
    row_index = 0
    col_index = 0

    '''initilize simlarity matrix'''
    sim_matrix = np.zeros((twi_js_file_linenum, flickr_js_file_linenum))

    twi_f = open(twitter_js_filepath)
    flickr_f = open(flickr_js_filepath)

    twi_lines = []
    flickr_lines = []

    for twi_line in twi_f:
        twi_lines.append(twi_line)

    for flickr_line in flickr_f:
        flickr_lines.append(flickr_line)

    for twi_line in twi_lines:
        for flickr_line in flickr_lines:

            try:
                twi_info_js = json.loads(twi_line)
                flickr_info_js = json.loads(flickr_line)
            except ValueError:
                continue

            flickr_owner_id = flickr_info_js['owner_id']
            twi_name = twi_info_js['screen_name']

            flickr_profilepic_path = flickr_images_root_dir + flickr_owner_id + '/profile_image.jpg'
            twi_profilepic_path = twi_images_root_dir + twi_name + '/profile_image.jpg'



            try:
                # both profile pics have faces, store them for future analysis
                if FaceDetection.face_detect(flickr_profilepic_path, cascPath) > 0 \
                        and FaceDetection.face_detect(twi_profilepic_path, cascPath) > 0:
                    print 'Both have faces'
                    record_f = open(both_face_txt_path, 'a')
                    record_f.write(str(row_index) + ', ' + str(col_index)
                                   + ', ' + twi_name + ', ' + flickr_owner_id)
                    record_f.write('\n')
                    record_f.close()

                elif FaceDetection.face_detect(flickr_profilepic_path, cascPath) == 0\
                        and FaceDetection.face_detect(twi_profilepic_path, cascPath) > 0:
                    print 'only one has faces'
                    sim_matrix[row_index][col_index] = 0.0

                elif FaceDetection.face_detect(flickr_profilepic_path, cascPath) > 0\
                        and FaceDetection.face_detect(twi_profilepic_path, cascPath) == 0:
                    print 'only one has faces'
                    sim_matrix[row_index][col_index] = 0.0

                else:
                    print "Both don't have faces, do sift matching"
                    kp1, kp2, matches, sim_score = image_matching.match_images(flickr_profilepic_path, twi_profilepic_path)
                    sim_matrix[row_index][col_index] = sim_score

            except:
                sim_matrix[row_index][col_index] = 0.0

            col_index += 1

        col_index = 0
        row_index += 1

    # saving matrix
    # later use np.loadtxt(txt_path) to load the
    print 'Finished comparing all the pictures, saving the similarity_matrix.'
    np.savetxt(save_txt_path, sim_matrix)
    return sim_matrix


def profile_img_similarity_face(target_root, both_faces_filepath, save_txt_path, aligned_size, matrix_rownum, matrix_colnum):
    face_pair_info = dict()
    face_pair_infos = []
    with open(both_faces_filepath) as both_face_f:
        for line in both_face_f:
            line_parts = line.replace('\n', '').split(',')
            face_pair_info['row_index'] = int(line_parts[0].replace(' ', ''))
            face_pair_info['col_index'] = int(line_parts[1].replace(' ', ''))
            face_pair_info['twi_face_user'] = line_parts[2].replace(' ', '')
            face_pair_info['flickr_face_user'] = line_parts[3].replace(' ', '')
            face_pair_infos.append(face_pair_info.copy())
            face_pair_info.clear()

    # copying faces dir to aligned dir
    print 'copying faces data...'
    twi_faces_dir = target_root + 'twitter_faces/'
    flickr_faces_dir = target_root + 'flickr_faces/'
    if not os.path.exists(twi_faces_dir):
        os.makedirs(twi_faces_dir)
    if not os.path.exists(flickr_faces_dir):
        os.makedirs(flickr_faces_dir)

    for face_pair_info in face_pair_infos:
        if not os.path.exists(twi_faces_dir + face_pair_info['twi_face_user'] + '/'):
            os.makedirs(twi_faces_dir + face_pair_info['twi_face_user'] + '/')
            os.system('cp -a ' + target_root + 'twitter_profile_images/' + face_pair_info['twi_face_user'] + '/*'
                  + ' ' + twi_faces_dir + face_pair_info['twi_face_user'] + '/')

        if not os.path.exists(flickr_faces_dir + face_pair_info['flickr_face_user'] + '/'):
            os.makedirs(flickr_faces_dir + face_pair_info['flickr_face_user'] + '/')
            os.system('cp -a ' + target_root + 'flickr_profile_images/' + face_pair_info['flickr_face_user'] + '/*'
                  + ' ' + flickr_faces_dir + face_pair_info['flickr_face_user'] + '/')

    # print 'aligning faces...'
    # twi_aligned_faces_dir = target_root + 'twitter_faces_' + str(aligned_size)
    # flk_aligned_faces_dir = target_root + 'flickr_faces_' + str(aligned_size)
    #
    # os.system('python ../facenet/src/align_dataset_mtcnn.py ' + target_root + 'twitter_faces/ ' +
    #           twi_aligned_faces_dir + ' --image_size ' + str(aligned_size) + ' --margin 44')
    #
    # os.system('python ../facenet/src/align_dataset_mtcnn.py ' + target_root + 'flickr_faces/ ' +
    #           flk_aligned_faces_dir + ' --image_size ' + str(aligned_size) + ' --margin 44')

    print 'matching faces...'
    sim_matrix = np.zeros((matrix_rownum, matrix_colnum))
    for face_pair_info in face_pair_infos:
        print 'print matching %s and %s at (%d, %d)' % (face_pair_info['twi_face_user'], face_pair_info['flickr_face_user'],
                                                        face_pair_info['row_index'], face_pair_info['col_index'])
        try:
            twi_pic_path = target_root + 'twitter_faces_' + str(aligned_size) + '/' + \
                           face_pair_info['twi_face_user'] + '/profile_image.png'


            flk_pic_path = target_root + 'flickr_faces_' + str(aligned_size) + '/' + \
                           face_pair_info['flickr_face_user'] + '/profile_image.png'

            sim_score = FaceDetection.FaceMatching(twi_pic_path, flk_pic_path)

        except (TypeError, IOError):
            try:
                twi_pic_path = target_root + 'twitter_profile_images/' + \
                               face_pair_info['twi_face_user'] + '/profile_image.jpg'

                flk_pic_path = target_root + 'flickr_profile_images/' + \
                               face_pair_info['flickr_face_user'] + '/profile_image.jpg'

                __, __, __, sim_score = image_matching.match_images(twi_pic_path, flk_pic_path)

            except Exception:
                sim_score = 0.0

        sim_matrix[face_pair_info['row_index']][face_pair_info['col_index']] = sim_score
        np.savetxt(save_txt_path, sim_matrix)


if __name__ == '__main__':

    # personal_website_similarity('../target_data/twitter_info_list.js',
    #                             '../target_data/flickr_info_list.js')
    #
    # username_similarity('../target_data/twitter_info_list.js',
    #                     '../target_data/flickr_info_list.js')

    # location_similaritydef('../target_data/twitter_info_list.js',
    #                 '../target_data/flickr_info_list.js')

      # profile_img_similarity_no_face('../target_data/twitter_info_list.js',
      #           '../target_data/flickr_info_list.js', '../target_data/twitter_profile_images/',
      #                                '../target_data/flickr_profile_images/','../target_data/profile_pic_similarity_matrix.txt'
      #                                , '../target_data/both_faces.txt')

    profile_img_similarity_face('../target_data/',
                                '../target_data/both_have_faces_pairs.txt',
                                '../target_data/profile_pic_similarity_matrix_face.txt', 160, 2068, 504)
    # test strings similarity
    # string1 = 'my name is yinhao'
    # string2 = 'something is common'
    #
    # string1_list = string1.split(' ')
    # string2_list = string2.split(' ')
    #
    # print calc_stringlist_similarity_LSA(string1_list, string2_list)

    ## get bio first
    # with open('../target_data/flickr_info_list.js') as f:
    #     for line in f:
    #         info = json.loads(line)
    #
    #         r = requests.get('https://www.flickr.com/people/' + info['owner_id'] + '/')
    #         soup = BeautifulSoup(r.text.encode('utf-8'), 'lxml')
    #         bio = soup.find('meta', {'property': 'twitter:description'})
    #
    #         if bio is not None:
    #             info['bio'] = bio['content']
    #
    #         f_out = open('../target_data/flickr_info_list_bioadded.js', 'a')
    #         json.dump(info, f_out)
    #         f_out.write('\n')
    #         f_out.close()
    #
    #         print 'Dealing with user ' + info['ownername']

