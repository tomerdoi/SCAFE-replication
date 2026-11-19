"""Use candidate-only cached exclusion flags for intake; never resolve bindings."""
from __future__ import annotations
from collections import Counter
from map_function_scopes import scope_index

QUALIFIERS = {'external','public','internal','private','pure','view','payable','virtual','constant'}

def modifier_shapes(fragment):
    """Return syntactic modifier-shaped header names, not compiler references."""
    tokens = scope_index(fragment)['tokens']
    values = [t[1] for t in tokens]
    if len(values)<4 or values[0]!='function' or tokens[1][0]!='identifier' or values[2]!='(':
        raise ValueError('Expected direct named implemented function fragment')
    def skip_group(i):
        if i>=len(values) or values[i]!='(': raise ValueError('Missing argument/return group')
        depth=0
        for j in range(i,len(values)):
            if tokens[j][0]=='literal': continue
            if values[j]=='(': depth+=1
            elif values[j]==')':
                depth-=1
                if depth==0:return j+1
        raise ValueError('Unclosed header group')
    i=skip_group(2);names=[]
    while i<len(values):
        value=values[i]
        if value=='{':return names
        if value in QUALIFIERS:i+=1;continue
        if value in {'override','returns'}:
            i+=1
            if i<len(values) and values[i]=='(':i=skip_group(i)
            elif value=='returns':raise ValueError('Missing returns group')
            continue
        if tokens[i][0]!='identifier':raise ValueError('Unsupported header token')
        name=value;i+=1
        while i<len(values) and values[i]=='.':
            if i+1>=len(values) or tokens[i+1][0]!='identifier':raise ValueError('Malformed qualified modifier name')
            name+='.'+values[i+1];i+=2
        names.append(name)
        if i<len(values) and values[i]=='(':i=skip_group(i)
    raise ValueError('No function body observed')

def aggregate(rows):
    """Each row is a bound public-candidate member plus opaque prior hit kinds."""
    counts=Counter();known={'exact','tokens','template'}
    for row in rows:
        hits=set(row['hits'])
        if hits-known:raise ValueError('Unsupported cached reason')
        if row['header_status']!='parsed':
            counts['unknown_headers']+=1;continue
        counts['parsed_named_members']+=1
        mods=row['modifier_shapes'];counts['lexical_modifier_shapes']+=len(mods)
        if not mods:continue
        counts['query_shaped_members']+=1
        direct=bool(hits & {'exact','tokens'})
        if direct:
            counts['query_members_with_direct_content_match']+=1
            counts['lexical_shapes_in_direct_matched_members']+=len(mods)
        elif 'template' in hits:
            counts['query_members_with_template_only_match']+=1
        else:counts['query_members_with_no_recorded_match']+=1
    if counts['query_members_with_direct_content_match']:
        decision='exclude_current_unchanged_query_scope'
    elif counts['query_members_with_template_only_match']:
        decision='quarantine_current_unchanged_query_scope'
    else:decision='unknown_never_positive_admission'
    keys=['parsed_named_members','unknown_headers','query_shaped_members','lexical_modifier_shapes','query_members_with_direct_content_match','lexical_shapes_in_direct_matched_members','query_members_with_template_only_match','query_members_with_no_recorded_match']
    return {'counts':{k:counts[k] for k in keys},'decision':decision,'positive_source_admission':False,
            'compiler_reference_queries':'not measured; lexical header anchors only',
            'family_identity':'not established by a content match; no whole-family claim',
            'scope_policy':'Keep the original complete application-query scope; no filtered survivors, source edits or substitute variants.'}
