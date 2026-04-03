# # def adaptive_fusion(face_score, voice_score):
# #     """
# #     Improved fusion logic (research contribution)
# #     """

# #     # Trust face more when voice is weak
# #     def adaptive_fusion(face_score, voice_score):

# #     # stronger voice contribution
# #         if voice_score < 0.5:
# #             return 0.8 * face_score + 0.2 * voice_score
# #         else:
# #             return 0.6 * face_score + 0.4 * voice_score

# def adaptive_fusion(face_score, voice_score):
#     # Handle missing modalities
#     if face_score is None and voice_score is None:
#         return None

#     if face_score is None:
#         return voice_score

#     if voice_score is None:
#         return face_score

#     # Weighted fusion
#     return 0.6 * face_score + 0.4 * voice_score

def adaptive_fusion(face_score, voice_score):
    """
    Adaptive fusion (research novelty)
    """

    # Dynamic weighting
    if face_score > 0.8:
        w_face = 0.7
    else:
        w_face = 0.5

    w_voice = 1 - w_face

    final_score = w_face * face_score + w_voice * voice_score
    return final_score