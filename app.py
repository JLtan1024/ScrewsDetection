# Calculate pixel-to-mm ratio using 10sen coin (class 13)
            coin_detections = [det for det in obb_detections if len(det) == 7 and int(det[6]) == COIN_CLASS_ID]

            if coin_detections:
                coin = coin_detections[0]
                # Width and height are at indices 2 and 3
                if len(coin) >= 4:
                    width_px = coin[2]
                    height_px = coin[3]
                    avg_px_diameter = (width_px + height_px) / 2
                    if avg_px_diameter > 0:
                        px_to_mm_ratio = COIN_DIAMETER_MM / avg_px_diameter
                    else:
                        st.warning("Detected coin has zero diameter, cannot calculate ratio.")
                        px_to_mm_ratio = None
                else:
                    st.warning("Coin detection data is incomplete.")
                    px_to_mm_ratio = None

                if px_to_mm_ratio is not None:
                    screw_lengths = []
                    for det in obb_detections:
                        if len(det) == 7:
                            class_id = int(det[6])
                            # Width and height for all objects are at indices 2 and 3
                            if class_id != COIN_CLASS_ID:
                                width_px = det[2]
                                height_px = det[3]
                                length_px = max(width_px, height_px)
                                length_mm = length_px * px_to_mm_ratio
                                screw_lengths.append((class_id, length_mm))
                            elif class_id != COIN_CLASS_ID:
                                st.warning(f"Incomplete data for non-coin detection.")

                    # Display measurements
                    st.subheader("📏 Screw Measurements:")
                    if screw_lengths:
                        for class_id, length_mm in screw_lengths:
                            st.write(f"Class {class_id} screw/nut length: {length_mm:.2f} mm")
                    else:
                        st.warning("No screws/nuts detected (only coin found)")
            else:
                st.warning("No 10sen coin detected - cannot calculate measurements without reference")
