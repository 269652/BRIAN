(region Hippocampus
  (def layers 2)
  (def connections 'sequential)
  (def learning_rule 'hebbian)
  (def recall_topk 5)

  ;; Projections
  (def projections
    [ (projection 'Hippocampus 'PFC 'signal nil (lambda (ctx) true))
      (projection 'Hippocampus 'DMN 'signal nil (lambda (ctx) (> (get ctx 'recall_strength) 0.5)))
      (projection 'Hippocampus 'BG 'nt 'ACh (lambda (ctx) (> (get ctx 'novelty) 0.7)))
    ])

  ;; NT production
  (def nt_production
    [ (nt_prod 'ACh (lambda (ctx) (* 0.5 (get ctx 'recall_topk))) (lambda (ctx) true))
    ])

  (defun enrich (query)
    (let ((results (query-memory query recall_topk)))
      results))

  (defun on-epigenetic-signal (nt-state)
    (if (> (get nt-state 'ACh) 0.7)
  (set! recall_topk (+ recall_topk 1))))

)
