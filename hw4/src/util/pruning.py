

import re
from collections import defaultdict


def PruneDict():

    prune_dict = {}

    prune_dict['u']  = 'you'
    prune_dict['ur']  = 'your'
    prune_dict['dis'] = 'this'
    prune_dict['dat'] = 'that'
    prune_dict['gurl'] = 'girl'
    prune_dict['teh'] = prune_dict['da'] = prune_dict['tha'] = 'the'
    prune_dict['evar'] = 'ever'
    prune_dict['likes'] = prune_dict['liked'] = prune_dict['lk'] = 'like'
    prune_dict['wierd'] = 'weird'
    prune_dict['kool'] = 'cool'
    prune_dict['yess'] = 'yes'
    prune_dict['pleasee'] = 'please'
    prune_dict['soo'] = 'so'
    prune_dict['noo'] = 'no'
    prune_dict['lovee'] = prune_dict['loove'] = prune_dict['looove'] = prune_dict['loooove'] = prune_dict['looooove'] \
        = prune_dict['loooooove'] = prune_dict['loves'] = prune_dict['loved'] = prune_dict['wuv'] \
        = prune_dict['loovee'] = prune_dict['lurve'] = prune_dict['lov'] = prune_dict['luvs'] = 'love'
    prune_dict['lovelove'] = 'love love'
    prune_dict['lovelovelove'] = 'love love love'
    prune_dict['ilove'] = 'i love'
    prune_dict['liek'] = prune_dict['lyk'] = prune_dict['lik'] = prune_dict['lke'] = prune_dict['likee'] = 'like'
    prune_dict['mee'] = 'me'
    prune_dict['hooo'] = 'hoo'
    prune_dict['sooon'] = prune_dict['soooon'] = 'soon'
    prune_dict['goodd'] = prune_dict['gud'] = 'good'
    prune_dict['bedd'] = 'bed'
    prune_dict['badd'] = 'bad'
    prune_dict['sadd'] = 'sad'
    prune_dict['madd'] = 'mad'
    prune_dict['redd'] = 'red'
    prune_dict['tiredd'] = 'tired'
    prune_dict['boredd'] = 'bored'
    prune_dict['godd'] = 'god'
    prune_dict['xdd'] = 'xd'
    prune_dict['itt'] = 'it'
    prune_dict['lul'] = prune_dict['lool'] = 'lol'
    prune_dict['sista'] = 'sister'
    prune_dict['w00t'] = 'woot'
    prune_dict['srsly'] = 'seriously'
    prune_dict['4ever'] = prune_dict['4eva'] = prune_dict['fourever'] = prune_dict['foureva'] = prune_dict['foreva'] = 'forever'
    prune_dict['neva'] = 'never'
    prune_dict['2day'] = 'today'
    prune_dict['homee'] = 'home'
    prune_dict['hatee'] = 'hate'
    prune_dict['heree'] = 'here'
    prune_dict['cutee'] = 'cute'
    prune_dict['lemme'] = 'let me'
    prune_dict['mrng'] = 'morning'
    prune_dict['gd'] = 'good'
    prune_dict['thx'] = prune_dict['thnx'] = prune_dict['thanx'] = prune_dict['thankx'] = prune_dict['thnk'] = 'thanks'
    prune_dict['jaja'] = prune_dict['jajaja'] = prune_dict['jajajaja'] = 'haha'
    prune_dict['eff'] = prune_dict['fk'] = prune_dict['fuk'] = prune_dict['fuc'] = 'fuck'
    prune_dict['2moro'] = prune_dict['2mrow'] = prune_dict['2morow'] = prune_dict['2morrow'] \
        = prune_dict['2morro'] = prune_dict['2mrw'] = prune_dict['2moz'] = 'tomorrow'
    prune_dict['babee'] = 'babe'
    prune_dict['theree'] = 'there'
    prune_dict['thee'] = 'the'
    prune_dict['woho'] = prune_dict['wohoo'] = 'woo hoo'
    prune_dict['2gether'] = 'together'
    prune_dict['2nite'] = prune_dict['2night'] = 'tonight'
    prune_dict['nite'] = 'night'
    prune_dict['dnt'] = 'do not'
    prune_dict['rly'] = 'really'
    prune_dict['gt'] = 'get'
    prune_dict['lat'] = 'late'
    prune_dict['dam'] = 'damn'
    prune_dict['4ward'] = 'forward'
    prune_dict['4give'] = 'forgive'
    prune_dict['b4'] = 'before'
    prune_dict['tho'] = 'though'
    prune_dict['kno'] = 'know'
    prune_dict['grl'] = 'girl'
    prune_dict['boi'] = 'boy'
    prune_dict['wrk'] = 'work'
    prune_dict['jst'] = 'just'
    prune_dict['geting'] = 'getting'
    prune_dict['4get'] = 'forget'
    prune_dict['4got'] = 'forgot'
    prune_dict['4real'] = 'for real'
    prune_dict['2go'] = 'to go'
    prune_dict['2b'] = 'to be'
    prune_dict['gr8'] = prune_dict['gr8t'] = prune_dict['gr88'] = 'great'
    prune_dict['str8'] = 'straight'
    prune_dict['twiter'] = 'twitter'
    prune_dict['iloveyou'] = 'i love you'
    prune_dict['loveyou'] = prune_dict['loveya'] = prune_dict['loveu'] = 'love you'
    prune_dict['xoxox'] = prune_dict['xox'] = prune_dict['xoxoxo'] = prune_dict['xoxoxox'] \
        = prune_dict['xoxoxoxo'] = prune_dict['xoxoxoxoxo'] = 'xoxo'
    prune_dict['cuz'] = prune_dict['bcuz'] = prune_dict['becuz'] = 'because'
    prune_dict['iz'] = 'is'
    prune_dict['aint'] = 'am not'
    prune_dict['fav'] = 'favorite'
    prune_dict['ppl'] = 'people'
    prune_dict['mah'] = 'my'
    prune_dict['r8'] = 'rate'
    prune_dict['l8'] = 'late'
    prune_dict['w8'] = 'wait'
    prune_dict['m8'] = 'mate'
    prune_dict['h8'] = 'hate'
    prune_dict['l8ter'] = prune_dict['l8tr'] = prune_dict['l8r'] = 'later'
    prune_dict['cnt'] = 'cant'
    prune_dict['fone'] = prune_dict['phonee'] = 'phone'
    prune_dict['jammin'] = 'jamming'
    prune_dict['onee'] = 'one'
    prune_dict['1st'] = 'first'
    prune_dict['2nd'] = 'second'
    prune_dict['3rd'] = 'third'
    prune_dict['inet'] = 'internet'
    prune_dict['recomend'] = 'recommend'
    prune_dict['any1'] = 'anyone'
    prune_dict['every1'] = prune_dict['evry1'] = 'everyone'
    prune_dict['some1'] = prune_dict['sum1'] = 'someone'
    prune_dict['no1'] = 'no one'
    prune_dict['4u'] = 'for you'
    prune_dict['4me'] = 'for me'
    prune_dict['2u'] = 'to you'
    prune_dict['yu'] = 'you'
    prune_dict['yr'] = prune_dict['yrs'] = prune_dict['years'] = 'year'
    prune_dict['hr'] = prune_dict['hrs'] = prune_dict['hours'] = 'hour'
    prune_dict['min'] = prune_dict['mins'] = prune_dict['minutes'] = 'minute'
    prune_dict['go2'] = prune_dict['goto'] = 'go to'

    return prune_dict


def PrunePunctuation(word):

    q_count, e_count, d_count, c_count = word.count('?'), word.count('!'), word.count('.'), word.count(',')

    if q_count:
        return '_?'
    elif e_count >= 5:
        return '_!!!'
    elif e_count >= 3:
        return '_!!'
    elif e_count >= 1:
        return '_!'
    elif d_count >= 2:
        return '_...'
    elif d_count == 1:
        return '_.'
    elif c_count >= 1:
        return '_,'

    return word


def PruneNumber(word):

    if word == '2':
        return 'to'
    elif word == '4':
        return 'for'
    elif word == '1':
        return 'one'
    elif word.isnumeric():
        return '_num'

    return word


def PruneDuplicateCharacter(word, word_freq, word_freq_thres=30):

    # prune duplicate character at rear
    word = re.sub(r'(([a-z])\2{2,})$', r'\g<2>\g<2>', word)
    word = re.sub(r'(([a-cg-kmnp-ru-z])\2+)$', r'\g<2>', word)

    # prune duplicate character start from head
    word = re.sub(r'^(([a-km-z])\2+)', r'\g<2>', word)

    # prune duplicate character inline
    word = re.sub(r'(([a-z])\2{2,})', r'\g<2>\g<2>', word)
    word = re.sub(r'(([ahjkquvwxyz])\2+)', r'\g<2>', word)

    if word_freq[word] > word_freq_thres:
        return word
    else:

        further_pruned_word = re.sub(r'(([a-z])\2+)', r'\g<2>', word)

        if word_freq[word] >= word_freq[further_pruned_word]:
            return word
        else:
            return further_pruned_word


def PruneContraction(pre_word, post_word):

    if post_word == 't':

        if pre_word in ['don','didn','doesn','haven','isn','wasn','couldn','aren','wouldn',
                        'hasn','weren','hadn','shouldn']:
            pre_word, post_word = pre_word[:-1], "not"
        elif pre_word in ['can']:
            pre_word, post_word = pre_word, "not"
        elif pre_word == 'won':
            pre_word, post_word = "will", "not"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 's':

        if pre_word in ['it','that','he','she','there','what','how','who','here','where']:
            pre_word, post_word = pre_word, "is"
        elif pre_word == 'let':
            pre_word, post_word = "let", "us"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 'll':

        if pre_word in ['i','you','we','it','they','he','she']:
            pre_word, post_word = pre_word, "will"
        elif pre_word == 'ya':
            pre_word, post_word = "you", "all"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 'm':

        if pre_word == 'i':
            pre_word, post_word = pre_word, "am"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 're':

        if pre_word in ['you','we', 'they']:
            pre_word, post_word = pre_word, "are"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 've':

        if pre_word in ['i','you','we', 'they']:
            pre_word, post_word = pre_word, "have"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    elif post_word == 'all':

        if pre_word == 'ya':
            pre_word, post_word = "you", "all"
        else:
            pre_word, post_word = pre_word, "\'" + post_word

    else:
        pre_word, post_word = pre_word, "\'" + post_word

    return pre_word, post_word


def PruneCategory(word):

    mch = re.match(r'^[0-9]+(.*)', word)

    if mch:
        suffix = mch.group(1)
    else:
        return word

    if suffix in ['st', 'nd', 'rd', 'th']:
        return '_order'
    elif suffix in ['am','pm', 'min','mins', 'hr', 'hrs', 'h', 'hour', 'hours', 'yr', 'yrs', 'day', 'days', 'wks']:
        return '_time'
    else:
        return word


def PruneRareWord(text, word_freq, freq_thres=5):

    sen_list = []
    for word in text.split():
        if word_freq[word] < freq_thres:
            sen_list.append('_rare')
        else:
            sen_list.append(word)

    return ' '.join(sen_list)
