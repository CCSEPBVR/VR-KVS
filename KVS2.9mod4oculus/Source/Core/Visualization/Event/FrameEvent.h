/*****************************************************************************/
/**
 *  @file   FrameEvent.h
 */
/*----------------------------------------------------------------------------
 *
 *  Copyright (c) Visualization Laboratory, Kyoto University.
 *  All rights reserved.
 *  See http://www.viz.media.kyoto-u.ac.jp/kvs/copyright/ for details.
 *
 *  $Id$
 */
/*****************************************************************************/
#ifndef KVS__FRAME_EVENT_H_INCLUDE
#define KVS__FRAME_EVENT_H_INCLUDE

#include <kvs/EventBase>


namespace kvs
{

/*===========================================================================*/
/**
 *  @brief  PaintEvent class.
 */
/*===========================================================================*/
class FrameEvent : public kvs::EventBase
{
public:

    FrameEvent();
    virtual ~FrameEvent();

    int type() const;
};

} // end of namespace kvs

#endif // KVS__FRAME_EVENT_H_INCLUDE
